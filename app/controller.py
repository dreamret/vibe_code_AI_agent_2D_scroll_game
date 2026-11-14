"""Application controller: wires capture, perception, input, learning, router and DB.

Contains no Qt code - the GUI registers plain callbacks (which it turns into Qt
signals) so every module stays independently testable.
"""
from __future__ import annotations

import hashlib
import threading
import time
from collections import deque
from pathlib import Path

import re
import cv2

from . import autolabel, gpu, media, winutil
from .imgio import imwrite
from .action_router import ActionRouter, validate_control
from .capture import CaptureService
from .config import Settings
from .db import Database, new_id, now_iso
from .input_recorder import InputRecorder
from .learning import (OutcomeTracker, StateEncoder, deploy_model, load_model,
                       train_candidate)
from .logbus import LogSystem
from .perception import PerceptionService
from .profiles import ProfileManager


def safe_label(label: str) -> str:
    """Labels are folder names: drop path/illegal characters, no leading '_' (reserved)."""
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", label.strip()).strip(". ")
    s = s.lstrip("_") or "unnamed"
    return s[:60]


class AppController:
    def __init__(self):
        self.settings = Settings.load()
        self.settings.ensure_dirs()
        self.log = LogSystem(self.settings)
        self.db = Database(self.settings, self.log)
        self.log.attach_store(self.db)
        self.db.purge_logs(int(self.settings["log_retention_days"]))

        gpu.accel.configure(bool(self.settings["gpu_opencl"]), self.log)
        threading.Thread(target=self._log_gpu_report, daemon=True, name="gpu-report").start()

        self.profiles = ProfileManager(self.db, self.log)
        self.profile: dict = self.profiles.ensure_default()
        self.target: dict | None = None

        self.capture = CaptureService(self.settings, self.log)
        self.perception = PerceptionService(self.settings, self.log)
        self.router = ActionRouter(self.settings, self.log, lambda: self.target,
                                   lambda: self.profile)
        self.recorder = InputRecorder(self.settings, self.log, lambda: self.target,
                                      self._on_human_action)
        self.outcomes = OutcomeTracker(self.db, float(self.settings["outcome_window_sec"]))

        # GUI callbacks (set by the GUI; called from worker threads)
        self.on_proposal = lambda p: None
        self.on_action = lambda a: None
        self.on_notice = lambda title, msg: None
        self.on_state = lambda: None
        self.on_autolabel = lambda clusters, source: None

        self.session: dict | None = None          # recording session
        self.session_dir: Path | None = None
        self.learn_enabled = False
        self.assist_enabled = False
        self.assist_session: dict | None = None
        self.recent_actions: deque = deque(maxlen=5)
        self.proposal: dict | None = None
        self._proposal_cooldown = 0.0
        self.policy = None
        self._scene_hist: deque = deque(maxlen=120)
        self._captcha_active = False
        self._captcha_ignore_until = 0.0
        self._replay_thread: threading.Thread | None = None
        self._replay_stop = threading.Event()
        self.replay_active = False
        self._unknown_since: float | None = None
        self._resume_cancel = threading.Event()
        self._resume_thread: threading.Thread | None = None
        self.resume_left = 0                      # seconds left in the get-ready countdown
        self._assist_stop = threading.Event()
        self._assist_thread: threading.Thread | None = None

        self.capture.subscribe(self.perception.submit)
        self.perception.subscribe(self._on_result)
        self.router.listeners.append(lambda _s: self.on_state())
        self.perception.configure(self.profile)
        self._load_policy()
        self.perception.start()
        threading.Thread(target=self.migrate_labels, daemon=True, name="migrate").start()
        self.router.start_hotkey()
        self.log.info("app", "Application ready", event_id="APP_READY",
                      database=self.db.backend_name, windows_platform=winutil.IS_WINDOWS)
        if not winutil.IS_WINDOWS:
            self.log.warning("app", "Window selection, focus checks and input execution are "
                             "implemented for Windows. Other platforms: whole-screen capture "
                             "only; automation will refuse to run.", event_id="PLATFORM")

    def _log_gpu_report(self):
        try:
            rep = gpu.gpu_report(self.settings)
            self.log.info("gpu", "GPU features: capture(dxcam)=%s opencl=%s directml=%s amf=%s" % (
                rep["dxcam"]["available"], rep["opencl"]["enabled"], rep["onnx"]["directml"],
                rep["ffmpeg"]["amf"]), event_id="GPU_REPORT")
        except Exception as exc:
            self.log.warning("gpu", f"GPU check failed: {exc}", event_id="GPU_REPORT_FAIL")

    def gpu_report_text(self) -> str:
        return gpu.report_text(gpu.gpu_report(self.settings), self.capture.backend_name
                               if self.capture.running else "")

    # ---- targets & profiles -------------------------------------------
    def list_windows(self) -> list[dict]:
        return winutil.list_windows()

    def find_window_for_profile(self) -> dict | None:
        g = self.profile.get("game", {})
        title, proc = g.get("window_title_contains", "").lower(), g.get("process_name", "").lower()
        for w in self.list_windows():
            if (title and title in w["title"].lower()) or (proc and proc == w["process"].lower()):
                return w
        return None

    def select_target(self, target: dict | None):
        self.target = target
        self.log.info("app", f"Target selected: {target['title'] if target else 'primary monitor'}",
                      event_id="TARGET_SELECTED", process=(target or {}).get("process"))
        if self.capture.running:
            self.capture.stop()
            self.capture.start(self.target)

    def select_profile(self, pid: str):
        p = self.profiles.get(pid)
        if not p:
            return
        self.profile = p
        self.perception.configure(p)
        self._load_policy()
        self.log.info("app", f"Profile selected: {p['name']}", event_id="PROFILE_SELECTED")
        self.on_state()

    def reload_profile(self):
        self.profile = self.profiles.get(self.profile["_id"])
        self.perception.configure(self.profile)
        self._load_policy()

    def _load_policy(self):
        self.policy = None
        mid = self.profile.get("active_model_id")
        if mid:
            m = self.db.get("models", mid)
            if m:
                self.policy = load_model(m["file_ref"])
                if self.policy:
                    self.log.info("learning", f"Active policy v{m['version']} loaded",
                                  event_id="POLICY_LOADED")

    # ---- capture -------------------------------------------------------
    def start_capture(self):
        self.capture.start(self.target)
        self.on_state()

    def stop_capture(self):
        if self.session:
            self.stop_recording()
        self.capture.stop()
        self.on_state()

    # ---- recording sessions -------------------------------------------
    def start_recording(self) -> bool:
        if not self.capture.running:
            self.log.warning("app", "Start capture before recording", event_id="REC_NO_CAPTURE")
            return False
        sid = new_id()
        self.session_dir = self.settings.path("recordings_dir") / sid
        self.session = {
            "_id": sid, "kind": "human",
            "application": {"title": (self.target or {}).get("title", "screen"),
                            "process": (self.target or {}).get("process", "")},
            "start_time": time.time(), "end_time": None, "profile_id": self.profile["_id"],
            "capture_settings": {"fps": self.settings["capture_fps"],
                                 "scale": self.settings["capture_scale"],
                                 "size": list(self.capture.stats["size"])},
            "status": "recording", "learn": self.learn_enabled,
            "video_ref": str(self.session_dir / "video.mp4"),
        }
        self.db.insert("sessions", self.session)
        self.capture.start_recording(self.session_dir)
        self.recorder.start()
        self.recent_actions.clear()
        self.log.info("app", f"Recording session {sid[:8]} started", event_id="SESSION_START",
                      learn=self.learn_enabled)
        self.on_state()
        return True

    def stop_recording(self):
        """Stops the recording; returns the finished session id (or None)."""
        if not self.session:
            return None
        sid = self.session["_id"]
        self.recorder.stop()
        self.capture.stop_recording()
        n_actions = self.db.count("actions", {"session_id": self.session["_id"]})
        self.db.update("sessions", self.session["_id"],
                       {"end_time": time.time(), "status": "completed",
                        "stats": {"actions": n_actions, "frames": self.capture.stats["frames"],
                                  "drops": self.capture.stats["drops"]}})
        self.log.info("app", f"Session {self.session['_id'][:8]} saved ({n_actions} actions)",
                      event_id="SESSION_END", drops=self.capture.stats["drops"])
        self.session = None
        self.db.flush()
        self.on_state()
        return sid

    def set_learn(self, on: bool):
        self.learn_enabled = on
        self.log.info("learning", f"Learning mode {'ON' if on else 'OFF'}", event_id="LEARN_TOGGLE")

    # ---- perception results -------------------------------------------
    def _on_result(self, res):
        self._scene_hist.append((res.timestamp, res.frame_index, res.scene))
        self.outcomes.update(res.timestamp, res.scene)

        # CAPTCHA: detect -> pause -> notify. Never solved automatically.
        if res.captcha.get("detected") and not self._captcha_active \
                and time.time() > self._captcha_ignore_until:
            self._captcha_active = True
            self.router.pause("captcha")
            self.log.warning("perception", f"CAPTCHA detected: {res.captcha.get('reason')}",
                             event_id="CAPTCHA_DETECTED", confidence=res.captcha.get("confidence"))
            if self.router.enabled or self.replay_active or self.assist_enabled:
                self.on_notice("CAPTCHA detected",
                               "Automation is paused. Please complete the challenge yourself, "
                               "then press 'Resume'.\n\nThis application never solves or bypasses "
                               "CAPTCHAs.")

        # watchdog while a replay is running
        if self.replay_active and not self.router.paused_reason:
            if not self.capture.ok:
                self.router.pause("capture_loss")
            if res.scene.get("scene_type") == "unknown":
                self._unknown_since = self._unknown_since or time.time()
                if time.time() - self._unknown_since > float(self.settings["unknown_state_timeout_sec"]):
                    self.router.pause("unknown_state")
            else:
                self._unknown_since = None

        # persist sampled frames + detections while recording
        if self.session and self.session_dir and \
                res.frame_index % max(1, int(self.settings["save_frame_every_n"])) == 0:
            try:
                self._save_frame(res)
            except Exception as exc:
                self.log.error("app", f"Frame save failed: {exc}", event_id="FRAME_SAVE_FAIL")

    def _save_frame(self, res):
        d = self.session_dir / "frames"
        d.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".jpg", res.image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        data = buf.tobytes()
        path = d / f"{res.frame_index:07d}.jpg"
        path.write_bytes(data)
        fid = self.db.insert("frames", {
            "session_id": self.session["_id"], "timestamp": res.timestamp,
            "frame_index": res.frame_index, "image_ref": str(path),
            "sha1": hashlib.sha1(data).hexdigest(), "scene_state": res.scene,
            "class_inference": res.class_info, "captcha": res.captcha,
            "detector_results": [e.to_dict() for e in res.entities]})
        if res.entities:
            self.save_labels([self.label_doc(
                e.label, e.bbox, "detector", session_id=self.session["_id"], frame_id=fid,
                frame_index=res.frame_index, timestamp=res.timestamp, confidence=e.confidence,
                track_id=e.track_id, name=e.name, detector=e.source) for e in res.entities])

    # ---- human actions -> DB + training examples ----------------------
    def _scene_at(self, ts: float):
        best = None
        for t, idx, scene in self._scene_hist:
            if t <= ts:
                best = (t, idx, scene)
        return best or (self._scene_hist[-1] if self._scene_hist else None)

    def _on_human_action(self, act: dict):
        if not self.session:
            return
        act["session_id"] = self.session["_id"]
        snap = self._scene_at(act["timestamp"])
        act["state_ref"] = {"frame_index": snap[1]} if snap else None
        act["_id"] = new_id()
        self.db.insert("actions", act)
        ctx = list(self.recent_actions)
        self.recent_actions.append(act["control"])
        if self.learn_enabled and snap:
            enc = StateEncoder(self.profile["detector"]["labels"])
            ex_id = self.db.insert("training_examples", {
                "profile_id": self.profile["_id"], "source_session": self.session["_id"],
                "kind": "demonstration", "timestamp": act["timestamp"],
                "state_features": enc.encode(snap[2]),
                "feature_labels": list(enc.labels),
                "state": {"counts": snap[2]["counts"], "scene_type": snap[2]["scene_type"]},
                "action": {"control": act["control"], "duration": act["duration"],
                           "input_device": act["input_device"]},
                "context_actions": ctx, "outcome": None, "labels": [],
                "frame_ref": snap[1], "review_status": "pending", "created": now_iso()})
            self.outcomes.add(ex_id, snap[0], snap[2])
        self.on_action(act)

    # ---- replay (human-approved) --------------------------------------
    def list_sessions(self) -> list[dict]:
        return [s for s in self.db.find("sessions", sort=("start_time", -1))
                if s.get("kind") == "human" and s.get("status") == "completed"]

    def replay_session(self, session_id: str, speed: float = 1.0, countdown: int = 5) -> bool:
        if self.replay_active:
            return False
        if not self.router.enabled:
            self.on_notice("Automation is OFF",
                           "Enable 'Input execution' in the Assist/Replay menu first.")
            return False
        acts = self.db.find("actions", {"session_id": session_id, "origin": "human"},
                            sort=("timestamp", 1))
        if not acts:
            self.on_notice("Nothing to replay", "This session has no recorded actions.")
            return False
        self._replay_stop.clear()
        self.replay_active = True
        self._unknown_since = None
        self._replay_thread = threading.Thread(target=self._replay_run,
                                               args=(session_id, acts, speed, countdown),
                                               daemon=True, name="replay")
        self._replay_thread.start()
        self.on_state()
        return True

    def stop_replay(self):
        self._replay_stop.set()

    def _replay_run(self, source_id, acts, speed, countdown):
        rs_id = new_id()
        self.db.insert("sessions", {"_id": rs_id, "kind": "replay", "source_session": source_id,
                                    "profile_id": self.profile["_id"], "start_time": time.time(),
                                    "end_time": None, "status": "replaying",
                                    "application": {"title": (self.target or {}).get("title", "")}})
        self.log.info("replay", f"Replay starts in {countdown}s - focus the game window now",
                      event_id="REPLAY_COUNTDOWN", actions=len(acts))
        for _ in range(countdown * 10):
            if self._replay_stop.is_set():
                break
            time.sleep(0.1)
        base = acts[0]["timestamp"]
        start = time.time()
        paused_total = 0.0
        delivered, failed, errs = 0, 0, []
        try:
            for a in acts:
                if self._replay_stop.is_set() or self.router.emergency:
                    break
                due = start + paused_total + (a["timestamp"] - base) / speed
                while time.time() < due and not self._replay_stop.is_set():
                    time.sleep(0.005)
                tries = 0
                res = {"ok": False, "reason": "aborted"}
                while not self._replay_stop.is_set() and not self.router.emergency:
                    if self.router.paused_reason:                 # wait for the human
                        t0 = time.time()
                        while self.router.paused_reason and not self._replay_stop.is_set() \
                                and not self.router.emergency:
                            time.sleep(0.1)
                        paused_total += time.time() - t0
                        self._unknown_since = None
                    req = {k: a[k] for k in ("input_device", "control", "duration", "x", "y")
                           if k in a}
                    res = self.router.execute(req, "replay")
                    if res["ok"]:
                        delivered += 1
                        errs.append(abs(res["delivered_at"] - max(due, start)) * 1000)
                        break
                    reason = res["reason"]
                    if reason in ("target_window_not_focused", "target_window_missing"):
                        self.router.pause("focus_lost" if reason.endswith("focused") else "window_lost")
                        continue
                    if reason == "rate_limit" and tries < 5:
                        tries += 1
                        time.sleep(0.12)
                        continue
                    failed += 1
                    break
                if res.get("reason") == "aborted":
                    break
                self.db.insert("actions", {
                    "session_id": rs_id, "origin": "replay", "source_action_id": a["_id"],
                    "timestamp": time.time(), "input_device": a["input_device"],
                    "control": a["control"], "value": "press", "duration": a["duration"],
                    "target_window": (self.target or {}).get("title", ""),
                    "execution_result": res})
        finally:
            self.replay_active = False
            self.db.update("sessions", rs_id, {
                "end_time": time.time(), "status": "completed",
                "stats": {"requested": len(acts), "delivered": delivered, "failed": failed}})
            mean_err = sum(errs) / len(errs) if errs else 0.0
            self.log.info("replay", f"Replay finished: {delivered}/{len(acts)} delivered, "
                          f"{failed} failed, mean timing error {mean_err:.0f} ms",
                          event_id="REPLAY_DONE", delivered=delivered, failed=failed,
                          mean_timing_error_ms=round(mean_err, 1))
            self.db.flush()
            self.on_state()

    # ---- assist mode (propose -> human approves) ----------------------
    def set_assist(self, on: bool):
        self.assist_enabled = on
        if on:
            if not self.policy:
                self.log.warning("assist", "No deployed model for this profile - train and "
                                 "deploy one in the Learning tab first.", event_id="NO_POLICY")
            self.assist_session = {"_id": new_id(), "kind": "assist", "start_time": time.time(),
                                   "profile_id": self.profile["_id"], "status": "assisting",
                                   "application": {"title": (self.target or {}).get("title", "")}}
            self.db.insert("sessions", self.assist_session)
            self._assist_stop.clear()
            self._assist_thread = threading.Thread(target=self._assist_loop, daemon=True,
                                                   name="assist")
            self._assist_thread.start()
        else:
            self._assist_stop.set()
            self.proposal = None
            self.on_proposal(None)
            if self.assist_session:
                self.db.update("sessions", self.assist_session["_id"],
                               {"end_time": time.time(), "status": "completed"})
                self.assist_session = None
        self.log.info("assist", f"Assist mode {'ON' if on else 'OFF'}", event_id="ASSIST_TOGGLE")
        self.on_state()

    def _assist_loop(self):
        enc = StateEncoder(self.profile["detector"]["labels"])
        while not self._assist_stop.wait(0.4):
            if not (self.capture.running and self.policy):
                continue
            if self.router.emergency or self.router.paused_reason:
                continue
            res = self.perception.latest()
            if res is None or time.time() - res.timestamp > 1.5:
                continue
            if self.proposal and time.time() - self.proposal["created"] < 4.0:
                continue
            if time.time() < self._proposal_cooldown:
                continue
            pred = self.policy.predict(res.scene, list(self.recent_actions))
            if pred and pred["confidence"] >= float(self.settings["min_confidence"]):
                self.proposal = {"id": new_id(), "created": time.time(), "action": pred,
                                 "state_features": enc.encode(res.scene),
                                 "feature_labels": list(enc.labels),
                                 "frame_index": res.frame_index,
                                 "state": {"counts": res.scene["counts"]}}
                self.log.info("assist", f"Proposed '{pred['control']}' "
                              f"(confidence {pred['confidence']:.2f})", event_id="PROPOSAL",
                              explanation=pred["explanation"][:3])
                self.on_proposal(self.proposal)
            elif self.proposal:
                self.proposal = None
                self.on_proposal(None)

    def approve_proposal(self):
        p, self.proposal = self.proposal, None
        if not p:
            return
        self.on_proposal(None)
        self._proposal_cooldown = time.time() + 1.0

        def run():
            a = p["action"]
            req = {"input_device": a["input_device"], "control": a["control"],
                   "duration": a["duration"]}
            res = self.router.execute(req, "assist")
            self.db.insert("actions", {
                "session_id": (self.assist_session or {}).get("_id", ""), "origin": "assist",
                "timestamp": time.time(), "value": "press", "target_window":
                (self.target or {}).get("title", ""), "state_ref": {"frame_index": p["frame_index"]},
                "execution_result": res, **req})
            if res["ok"]:
                self.recent_actions.append(a["control"])
            self.log.info("assist", f"Approved action '{a['control']}' -> "
                          f"{'sent' if res['ok'] else 'blocked: ' + res['reason']}",
                          event_id="PROPOSAL_APPROVED")
        threading.Thread(target=run, daemon=True).start()

    def reject_proposal(self, intended_control: str | None = None):
        p, self.proposal = self.proposal, None
        if not p:
            return
        self.on_proposal(None)
        self._proposal_cooldown = time.time() + 1.5
        intended = (intended_control or "").strip()
        self.db.insert("training_examples", {
            "profile_id": self.profile["_id"],
            "source_session": (self.assist_session or {}).get("_id", ""),
            "kind": "correction" if intended else "rejection", "timestamp": time.time(),
            "state_features": p["state_features"], "state": p["state"],
            "feature_labels": p.get("feature_labels", self.profile["detector"]["labels"]),
            "action": {"control": intended or p["action"]["control"], "duration": 0.1,
                       "input_device": "keyboard"},
            "rejected_action": p["action"]["control"], "context_actions": list(self.recent_actions),
            "outcome": None, "labels": ["user_correction"], "frame_ref": p["frame_index"],
            "review_status": "pending", "created": now_iso()})
        self.log.info("assist", f"Rejected '{p['action']['control']}'"
                      + (f"; intended '{intended}'" if intended else ""),
                      event_id="PROPOSAL_REJECTED")

    # ---- human controls -----------------------------------------------
    def resume(self):
        """Resume after a pause - but only after a get-ready countdown (default 5 s).
        Input stays blocked during the countdown."""
        if not self.router.paused_reason:
            return
        if self._resume_thread and self._resume_thread.is_alive():
            return                                # countdown already running
        self._resume_cancel.clear()
        self._resume_thread = threading.Thread(
            target=self._resume_run, args=(max(0, int(self.settings["resume_delay_sec"])),),
            daemon=True, name="resume")
        self._resume_thread.start()

    def cancel_resume(self):
        self._resume_cancel.set()

    def _resume_run(self, delay: int):
        if delay:
            self.log.info("router", f"Resuming in {delay} s - get ready (input still blocked)",
                          event_id="RESUME_COUNTDOWN")
        for left in range(delay, 0, -1):
            self.resume_left = left
            self.on_state()
            if self._resume_cancel.wait(1.0) or self.router.emergency:
                self.resume_left = 0
                self.log.info("router", "Resume cancelled", event_id="RESUME_CANCELLED")
                self.on_state()
                return
        self.resume_left = 0
        self._captcha_active = False
        self._captcha_ignore_until = time.time() + 10.0
        self._unknown_since = None
        self.router.resume()
        self.on_state()

    def emergency_stop(self):
        self.cancel_resume()
        self.router.emergency_stop()
        self._replay_stop.set()
        self.on_state()

    def stop_all(self):
        self.cancel_resume()
        self._replay_stop.set()
        if self.assist_enabled:
            self.set_assist(False)
        self.router.set_enabled(False)
        if self.session:
            self.stop_recording()
        self.capture.stop()
        self.on_state()

    # ---- corrections ---------------------------------------------------
    def add_object_correction(self, label: str, bbox: tuple, add_template: bool = True,
                              res=None) -> str:
        res = res or self.perception.latest()
        if res is None:
            raise RuntimeError("No frame available")
        h, w = res.image.shape[:2]
        x, y, bw, bh = [int(v) for v in bbox]
        x, y = max(0, x), max(0, y)
        bw, bh = min(bw, w - x), min(bh, h - y)
        if bw < 8 or bh < 8:
            raise ValueError("Selected box is too small")
        cdir = self.settings.path("corrections_dir")
        cdir.mkdir(parents=True, exist_ok=True)
        stamp = f"{int(time.time() * 1000)}"
        frame_path = cdir / f"frame_{stamp}.jpg"
        imwrite(frame_path, res.image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        label = safe_label(label)
        self.ensure_element(label, "other")
        eid = self.save_labels([self.label_doc(
            label, (x, y, bw, bh), "correction", session_id=(self.session or {}).get("_id", ""),
            frame_index=res.frame_index, timestamp=res.timestamp, frame_ref=str(frame_path))])[0]
        if add_template:
            label = safe_label(label)
            tdir = self.settings.path("templates_dir") / self.profile["_id"] / label
            tdir.mkdir(parents=True, exist_ok=True)
            imwrite(tdir / f"user_{stamp}.png", res.image[y:y + bh, x:x + bw])
            self.perception.reload_templates()
        self.log.info("correction", f"Labelled missed object as '{label}'",
                      event_id="OBJECT_CORRECTION", bbox=[x, y, bw, bh], template=add_template)
        return eid

    def annotate_situation(self, text: str) -> str:
        res = self.perception.latest()
        enc = StateEncoder(self.profile["detector"]["labels"])
        eid = self.db.insert("training_examples", {
            "profile_id": self.profile["_id"], "source_session": (self.session or {}).get("_id", ""),
            "kind": "annotation", "timestamp": time.time(), "note": text,
            "state_features": enc.encode(res.scene) if res else [],
            "feature_labels": list(enc.labels),
            "state": {"counts": res.scene["counts"]} if res else {},
            "action": {}, "context_actions": list(self.recent_actions), "outcome": None,
            "labels": ["annotation"], "review_status": "pending", "created": now_iso()})
        self.log.info("correction", "Situation annotated", event_id="ANNOTATION")
        return eid

    # ---- training review / models -------------------------------------
    def list_examples(self, status: str | None = None) -> list[dict]:
        q = {"profile_id": self.profile["_id"]}
        if status:
            q["review_status"] = status
        return self.db.find("training_examples", q, limit=2000, sort=("created", -1))

    def set_review(self, ids: list[str], status: str):
        for i in ids:
            self.db.update("training_examples", i, {"review_status": status})
        self.log.info("learning", f"{len(ids)} example(s) marked {status}", event_id="REVIEW")

    def pending_trainable(self) -> list[str]:
        """Ids of pending demonstrations/corrections (annotations and rejections
        are not used for training, so 'approve all' leaves them alone)."""
        return [e["_id"] for e in self.list_examples("pending")
                if e.get("kind", "demonstration") in ("demonstration", "correction")]

    def approve_all_pending(self) -> int:
        ids = self.pending_trainable()
        if ids:
            self.set_review(ids, "approved")
        return len(ids)

    def reject_all_pending(self) -> int:
        ids = self.pending_trainable()
        if ids:
            self.set_review(ids, "rejected")
        return len(ids)

    def delete_examples(self, ids: list[str]):
        for i in ids:
            self.db.delete("training_examples", {"_id": i})

    def list_models(self) -> list[dict]:
        return self.db.find("models", {"profile_id": self.profile["_id"]}, sort=("version", -1))

    def train(self) -> dict:
        return train_candidate(self.db, self.settings, self.log, self.profile)

    def deploy(self, model_id: str):
        self.profile = deploy_model(self.db, self.log, self.profile, model_id)
        self._load_policy()
        self.on_state()


    # ---- ONE label table: every label instance lives in `entities` ------
    # source = detector | correction | auto_label | media
    def label_doc(self, label: str, bbox, source: str, **extra) -> dict:
        doc = {"profile_id": self.profile["_id"], "label": label, "bbox": [int(v) for v in bbox],
               "confidence": 1.0, "track_id": -1, "source": source,
               "user_correction": source == "correction", "auto_label": source == "auto_label",
               "review_status": "approved" if source in ("correction", "auto_label") else "auto",
               "session_id": "", "media_id": "", "frame_index": -1, "timestamp": time.time()}
        doc.update(extra)
        return doc

    def save_labels(self, docs: list[dict]) -> list[str]:
        return self.db.insert_many("entities", docs)

    def migrate_labels(self):
        """Older entity rows: add `source` / `profile_id` so every label shares one schema."""
        try:
            sess = {x["_id"]: x.get("profile_id", "") for x in self.db.find("sessions")}
            fixed = 0
            for e in self.db.find("entities"):
                ch = {}
                if "source" not in e or e.get("source") in ("template", "yolo", "yolo-onnx"):
                    ch["source"] = ("correction" if e.get("user_correction") else
                                    "auto_label" if e.get("auto_label") else "detector")
                if "profile_id" not in e:
                    ch["profile_id"] = sess.get(e.get("session_id", ""), "")
                if ch:
                    self.db.update("entities", e["_id"], ch)
                    fixed += 1
            if fixed:
                self.log.info("database", f"Label table normalised ({fixed} rows)", event_id="LABELS_MIGRATED")
        except Exception as exc:
            self.log.warning("database", f"Label migration skipped: {exc}", event_id="LABELS_MIGRATE_FAIL")

    # ---- elements (user-configurable label catalogue) -------------------
    ELEMENT_CATEGORIES = ["character", "enemy", "npc", "item", "skill_effect", "ui", "marker",
                          "terrain", "text", "other"]

    def list_elements(self) -> list[dict]:
        return self.db.find("elements", {"profile_id": self.profile["_id"]}, sort=("label", 1))

    def element_template_count(self, label: str) -> int:
        d = self.settings.path("templates_dir") / self.profile["_id"] / label
        return len(list(d.glob("*.png")) + list(d.glob("*.jpg"))) if d.exists() else 0

    def save_element(self, data: dict) -> dict:
        data = dict(data)
        data["label"] = safe_label(data.get("label", ""))
        data["profile_id"] = self.profile["_id"]
        data.setdefault("display_name", data["label"])
        data.setdefault("category", "other")
        data.setdefault("enabled", True)
        existing = [e for e in self.list_elements() if e["label"] == data["label"]]
        if data.get("_id"):
            self.db.replace("elements", data["_id"], data)
        elif existing:
            data["_id"] = existing[0]["_id"]
            self.db.replace("elements", data["_id"], {**existing[0], **data})
        else:
            data["created"] = now_iso()
            data["_id"] = self.db.insert("elements", data)
        self.sync_labels()
        self.log.info("elements", f"Element '{data['label']}' saved", event_id="ELEMENT_SAVED",
                      category=data["category"])
        return data

    def delete_element(self, element_id: str, remove_templates: bool = False):
        el = self.db.get("elements", element_id)
        self.db.delete("elements", {"_id": element_id})
        if el and remove_templates:
            import shutil
            shutil.rmtree(self.settings.path("templates_dir") / self.profile["_id"] / el["label"],
                          ignore_errors=True)
            self.perception.reload_templates()
        self.log.info("elements", "Element deleted (the label stays in the profile until "
                      "you remove it there; stored examples are unaffected)", event_id="ELEMENT_DELETED")

    def ensure_element(self, label: str, category: str = "other") -> None:
        label = safe_label(label)
        if not any(e["label"] == label for e in self.list_elements()):
            self.save_element({"label": label, "category": category, "source": "auto_label"})

    def sync_labels(self):
        """Make every enabled element a detector label of the profile (append-only).
        Old training examples stay usable: features are re-mapped when training."""
        labels = list(self.profile["detector"]["labels"])
        added = [e["label"] for e in self.list_elements()
                 if e.get("enabled", True) and e["label"] not in labels]
        if added:
            self.profile["detector"]["labels"] = labels + added
            self.profile = self.profiles.save(self.profile)
            self.perception.configure(self.profile)
            self.log.info("elements", f"Added label(s) to profile: {', '.join(added)}",
                          event_id="LABELS_ADDED")
        self.on_state()

    def add_element_templates(self, label: str, files: list[str]) -> int:
        from .imgio import imread
        tdir = self.settings.path("templates_dir") / self.profile["_id"] / safe_label(label)
        tdir.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in files:
            img = imread(f)
            if img is not None:
                imwrite(tdir / f"import_{int(time.time() * 1000)}_{n}.png", img)
                n += 1
        self.perception.reload_templates()
        self.log.info("elements", f"Imported {n} template image(s) for '{label}'",
                      event_id="TEMPLATES_IMPORTED")
        return n

    # ---- database browser support --------------------------------------
    def after_db_edit(self, coll: str):
        if coll == "profiles":
            p = self.profiles.get(self.profile["_id"])
            if p is None:
                p = self.profiles.ensure_default()
            self.select_profile(p["_id"])
        elif coll == "elements":
            self.sync_labels()

    # ---- keybind comparison (profile vs. demo) --------------------------
    def keybind_hint(self, outcomes: list[dict]) -> str:
        if not outcomes:
            return "no outcome data (record with 'Learn' on to get hints)"
        n = len(outcomes)
        enemy = sum(o.get("enemy_delta", 0) < 0 for o in outcomes) / n
        item = sum(o.get("item_delta", 0) != 0 for o in outcomes) / n
        mx = sum(abs(o.get("player_dx", 0)) for o in outcomes) / n
        my = sum(o.get("player_dy", 0) for o in outcomes) / n
        parts = []
        if enemy >= 0.4:
            parts.append(f"enemies decreased after {enemy:.0%} of presses -> attack/skill?")
        if mx > 0.01:
            parts.append("player moved sideways -> movement?")
        if my < -0.02:
            parts.append("player moved up -> jump?")
        if item >= 0.4:
            parts.append("item count changed -> pick-up/interact?")
        return "; ".join(parts) or "no clear effect on the scene -> menu/utility key?"

    def analyze_keybinds(self, session_id: str) -> dict:
        prof = self.profile
        mapping = prof["input_mapping"]
        mapped = set(mapping.values())
        ignored = set(prof["constraints"].get("ignored_controls", []))
        acts = self.db.find("actions", {"session_id": session_id, "origin": "human"})
        used: dict[str, dict] = {}
        for a in acts:
            u = used.setdefault(a["control"], {"control": a["control"], "device": a["input_device"],
                                               "count": 0, "dur": 0.0})
            u["count"] += 1
            u["dur"] += float(a.get("duration", 0))
        exs = self.db.find("training_examples", {"source_session": session_id})
        unmapped = []
        for c, u in sorted(used.items(), key=lambda kv: -kv[1]["count"]):
            if c in mapped or c in ignored:
                continue
            outs = [e["outcome"] for e in exs if e.get("action", {}).get("control") == c
                    and e.get("outcome")]
            unmapped.append({"control": c, "device": u["device"], "count": u["count"],
                             "avg_duration": round(u["dur"] / u["count"], 3),
                             "hint": self.keybind_hint(outs)})
        return {"session_id": session_id, "unmapped": unmapped,
                "unused_mapped": sorted(n for n, c in mapping.items() if c not in used),
                "used_mapped": sorted(n for n, c in mapping.items() if c in used),
                "action_names": sorted(mapping)}

    # ---- keybinds: the profile's input_mapping IS the keybind table -----
    @staticmethod
    def device_of(control: str) -> str:
        return "mouse" if control.startswith("mouse_") else \
            "gamepad" if control.startswith("gamepad_") else "keyboard"

    def list_keybinds(self) -> list[dict]:
        pid = self.profile["_id"]
        meta = {k["control"]: k for k in self.db.find("keybinds", {"profile_id": pid})}
        rows = [{"action": n, "control": c, "device": self.device_of(c),
                 "purpose": meta.get(c, {}).get("purpose", ""), "status": "mapped"}
                for n, c in self.profile["input_mapping"].items()]
        for c in self.profile["constraints"].get("ignored_controls", []):
            rows.append({"action": "", "control": c, "device": self.device_of(c),
                         "purpose": meta.get(c, {}).get("purpose", ""), "status": "ignored"})
        return rows

    def set_keybind(self, action: str, control: str, purpose: str = "", old_action: str = "") -> str:
        """Add/rename/rebind a game action. Raises ValueError for bad or duplicate input."""
        control = validate_control(control)
        name = re.sub(r"\W+", "_", action.strip().lower()).strip("_")
        if not name:
            raise ValueError("Action name is empty")
        prof = self.profile
        mapping = prof["input_mapping"]
        for n, c in mapping.items():
            if c == control and n not in (name, old_action):
                raise ValueError(f"'{control}' is already used by '{n}'")
        if name != old_action and name in mapping:
            raise ValueError(f"Action '{name}' already exists")
        if old_action and old_action in mapping:
            mapping.pop(old_action)
        mapping[name] = control
        allowed = prof["constraints"].get("allowed_controls")
        if allowed and control not in allowed:
            allowed.append(control)
        ig = prof["constraints"].get("ignored_controls", [])
        if control in ig:
            ig.remove(control)
        self.profile = self.profiles.save(prof)
        pid = self.profile["_id"]
        ex = self.db.find("keybinds", {"profile_id": pid, "control": control})
        doc = {"profile_id": pid, "control": control, "device": self.device_of(control),
               "action_name": name, "purpose": purpose, "status": "mapped"}
        if ex:
            self.db.update("keybinds", ex[0]["_id"], doc)
        else:
            self.db.insert("keybinds", {**doc, "created": now_iso()})
        self.log.info("keybinds", f"Keybind {name} = {control}", event_id="KEYBIND_SET")
        self.on_state()
        return name

    def remove_keybind(self, action: str):
        prof = self.profile
        control = prof["input_mapping"].pop(action, None)
        if control is None:
            return
        allowed = prof["constraints"].get("allowed_controls")
        if allowed and control in allowed:
            allowed.remove(control)
        self.profile = self.profiles.save(prof)
        self.db.delete("keybinds", {"profile_id": prof["_id"], "control": control})
        self.log.info("keybinds", f"Keybind '{action}' removed", event_id="KEYBIND_REMOVED")
        self.on_state()

    def unignore_control(self, control: str):
        ig = self.profile["constraints"].get("ignored_controls", [])
        if control in ig:
            ig.remove(control)
            self.profile = self.profiles.save(self.profile)
            self.db.delete("keybinds", {"profile_id": self.profile["_id"], "control": control,
                                        "status": "ignored"})

    def test_keybind(self, control: str):
        """Send one press of `control` through the active input path (safety gates apply)."""
        def run():
            res = self.router.execute({"input_device": self.device_of(control), "control": control,
                                       "duration": 0.15, "x": 40, "y": 40}, "test")
            self.log.info("keybinds", f"Test of '{control}': "
                          + ("sent" if res["ok"] else f"blocked ({res['reason']})"),
                          event_id="KEYBIND_TEST", mode=self.router.mode)
        threading.Thread(target=run, daemon=True).start()

    def apply_keybind_answers(self, session_id: str, answers: list[dict]):
        """answers: {control, device, decision: add|ignore|later, action_name, purpose}"""
        prof = self.profile
        done = 0
        for a in answers:
            if a["decision"] == "later":
                continue
            if a["decision"] == "add":
                base = re.sub(r"\W+", "_", (a.get("action_name") or a.get("purpose") or a["control"])
                              .strip().lower()).strip("_") or a["control"]
                name, i = base, 2
                while name in self.profile["input_mapping"]:
                    name, i = f"{base}_{i}", i + 1
                try:
                    self.set_keybind(name, a["control"], a.get("purpose", ""))
                    done += 1
                except ValueError as exc:
                    self.log.warning("keybinds", f"Could not add {a['control']}: {exc}",
                                     event_id="KEYBIND_REJECTED")
            else:
                ig = self.profile["constraints"].setdefault("ignored_controls", [])
                if a["control"] not in ig:
                    ig.append(a["control"])
                self.profile = self.profiles.save(self.profile)
                self.db.insert("keybinds", {
                    "profile_id": self.profile["_id"], "control": a["control"],
                    "device": a.get("device", ""), "action_name": "", "purpose": a.get("purpose", ""),
                    "status": "ignored", "source_session": session_id, "created": now_iso()})
                done += 1
        self.log.info("keybinds", "Keybind answers saved to profile", event_id="KEYBINDS_SAVED",
                      count=done)
        self.on_state()

    # ---- auto-label ------------------------------------------------------
    def scan_live(self, seconds: float = 5.0, interval: float = 0.35) -> bool:
        if not self.capture.running:
            self.on_notice("Auto-label", "Start 'Capture' first.")
            return False

        def run():
            frames, ents, last = [], [], -1
            end = time.time() + seconds
            self.log.info("autolabel", f"Scanning the live capture for {seconds:.0f} s - move around "
                          "in the game so elements move", event_id="AUTOLABEL_START")
            while time.time() < end and len(frames) < 40:
                r = self.perception.latest()
                if r is not None and r.frame_index != last:
                    last = r.frame_index
                    frames.append(r.image.copy())
                    ents.append(list(r.entities))
                time.sleep(interval)
            lib = autolabel.load_library(self.settings.path("templates_dir"), self.profile["_id"])
            clusters = autolabel.scan_frames(frames, ents, lib) if frames else []
            self.log.info("autolabel", f"Found {len(clusters)} recurring unlabelled element group(s) "
                          f"in {len(frames)} frames", event_id="AUTOLABEL_DONE")
            self.on_autolabel(clusters, "live capture")
        threading.Thread(target=run, daemon=True, name="autolabel").start()
        return True

    def apply_clusters(self, selections: list[tuple], media_id: str = "") -> int:
        """selections: [(cluster, label)] - the human-approved labels."""
        troot = self.settings.path("templates_dir") / self.profile["_id"]
        n = 0
        for cl, label in selections:
            label = safe_label(label)
            self.ensure_element(label, cl.get("category", "other"))
            tdir = troot / label
            tdir.mkdir(parents=True, exist_ok=True)
            stamp = int(time.time() * 1000)
            for i, smp in enumerate(cl["samples"][:3]):
                imwrite(tdir / f"auto_{stamp}_{i}.png", smp["crop"])
            self.save_labels([self.label_doc(
                label, bb, "auto_label", media_id=media_id, frame_index=fi,
                confidence=cl.get("match_confidence") or 0.5) for fi, bb in cl["all_boxes"][:50]])
            n += 1
        self.perception.reload_templates()
        self.log.info("autolabel", f"Applied {n} label group(s)", event_id="AUTOLABEL_APPLIED")
        return n

    # ---- media (GIF / video / image) -------------------------------------
    def analyze_media(self, path: str, progress=None, stop=None):
        lib = autolabel.load_library(self.settings.path("templates_dir"), self.profile["_id"])
        self.log.info("media", f"Analysing {path}", event_id="MEDIA_START")
        res = media.analyze_media(path, self.perception, lib, progress, stop)
        res.media_id = self.db.insert("media", res.to_doc(self.profile["_id"]))
        rows = [self.label_doc(e.label, e.bbox, "media", media_id=res.media_id, frame_index=i,
                               confidence=e.confidence, track_id=e.track_id, name=e.name,
                               media_time=round(res.times[i], 3))
                for i, ents in enumerate(res.entities) for e in ents]
        if rows:
            self.save_labels(rows)
        self.log.info("media", "Media analysis saved to database", event_id="MEDIA_DONE",
                      frames=res.summary["frames_analyzed"], clusters=len(res.clusters),
                      labels_saved=len(rows))
        return res

    # ---- misc ----------------------------------------------------------
    def apply_settings(self):
        self.settings.save()
        self.outcomes.window = float(self.settings["outcome_window_sec"])
        self.log.info("app", "Settings saved (database/hotkey changes need a restart)",
                      event_id="SETTINGS_SAVED")

    def shutdown(self):
        self.log.info("app", "Shutting down", event_id="APP_SHUTDOWN")
        self.cancel_resume()
        self._replay_stop.set()
        self._assist_stop.set()
        self.router.set_enabled(False)
        if self.session:
            self.stop_recording()
        self.capture.stop()
        self.perception.stop()
        self.router.stop_hotkey()
        time.sleep(0.3)
        self.db.close()
        self.log.shutdown()
