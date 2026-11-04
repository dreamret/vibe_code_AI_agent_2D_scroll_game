"""Capture service: window/desktop-region capture, recording, health checks."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import winutil


@dataclass
class Frame:
    index: int
    timestamp: float
    image: np.ndarray            # BGR


class CaptureService:
    """Captures the client area of the selected window with `mss`.

    The window must be visible (not minimised / covered) for region capture.
    If no window is selected the primary monitor is captured. The backend is
    isolated in `_grab()` so a Windows Graphics Capture / DXGI backend can be
    plugged in without touching the rest of the application.
    """

    def __init__(self, settings, log):
        self.settings = settings
        self.log = log
        self.target: dict | None = None
        self.subscribers: list = []
        self.stats = {"frames": 0, "drops": 0, "failures": 0, "fps": 0.0,
                      "black_frames": 0, "size": (0, 0)}
        self.ok = False
        self.running = False
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._latest: Frame | None = None
        self._lock = threading.Lock()
        self._rec_lock = threading.Lock()
        self._writer = None
        self._rec_size = None
        self._rec_dir: Path | None = None
        self._ts_file = None
        self._last_warn = 0.0

    # ---- public --------------------------------------------------------
    def subscribe(self, cb):
        self.subscribers.append(cb)

    def latest(self) -> Frame | None:
        with self._lock:
            return self._latest

    def start(self, target: dict | None):
        if self.running:
            return
        self.target = target
        self._stop.clear()
        self.stats.update(frames=0, drops=0, failures=0, black_frames=0, fps=0.0)
        self._thread = threading.Thread(target=self._loop, daemon=True, name="capture")
        self.running = True
        self._thread.start()
        name = target["title"] if target else "primary monitor"
        self.log.info("capture", f"Capture started: {name}", event_id="CAPTURE_START",
                      fps=self.settings["capture_fps"])

    def stop(self):
        if not self.running:
            return
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        self.running = False
        self.ok = False
        self.stop_recording()
        self.log.info("capture", "Capture stopped", event_id="CAPTURE_STOP",
                      frames=self.stats["frames"], drops=self.stats["drops"])

    # ---- recording -----------------------------------------------------
    def start_recording(self, session_dir: Path):
        with self._rec_lock:
            self._rec_dir = Path(session_dir)
            self._rec_dir.mkdir(parents=True, exist_ok=True)
            self._ts_file = open(self._rec_dir / "frame_timestamps.csv", "w", encoding="utf-8")
            self._ts_file.write("frame_index,timestamp\n")
            self._writer = None   # created lazily on the first frame (needs size)
            self._rec_size = None
        self.log.info("capture", f"Recording to {session_dir}", event_id="REC_START")

    def stop_recording(self):
        with self._rec_lock:
            if self._writer is not None:
                self._writer.release()
            if self._ts_file:
                self._ts_file.close()
            self._writer = self._ts_file = self._rec_dir = self._rec_size = None

    def _record(self, frame: Frame, fps: float):
        with self._rec_lock:
            if self._rec_dir is None:
                return
            if self.settings["record_video"]:
                h, w = frame.image.shape[:2]
                if self._writer is None:
                    self._rec_size = (w, h)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    self._writer = cv2.VideoWriter(str(self._rec_dir / "video.mp4"),
                                                   fourcc, fps, self._rec_size)
                img = frame.image
                if (w, h) != self._rec_size:
                    img = cv2.resize(img, self._rec_size)
                self._writer.write(img)
            if self._ts_file:
                self._ts_file.write(f"{frame.index},{frame.timestamp:.4f}\n")

    # ---- capture loop --------------------------------------------------
    def _region(self):
        if self.target and self.target.get("hwnd"):
            hwnd = self.target["hwnd"]
            if not winutil.is_window(hwnd):
                return None
            r = winutil.client_rect(hwnd)
            if not r or r["width"] < 16 or r["height"] < 16:
                return None
            return r
        return "monitor"

    def _warn_throttled(self, msg, event_id):
        if time.time() - self._last_warn > 5:
            self._last_warn = time.time()
            self.log.warning("capture", msg, event_id=event_id)

    def _loop(self):
        try:
            import mss
        except ImportError:
            self.log.error("capture", "Package 'mss' is not installed", event_id="CAPTURE_DEP")
            self.running = False
            return
        index = 0
        black_run = 0
        fps_t0, fps_n = time.perf_counter(), 0
        with mss.mss() as sct:
            while not self._stop.is_set():
                fps = max(1, int(self.settings["capture_fps"]))
                interval = 1.0 / fps
                t0 = time.perf_counter()
                try:
                    region = self._region()
                    if region is None:
                        self.ok = False
                        self.stats["failures"] += 1
                        self._warn_throttled("Target window lost or minimised",
                                             "CAPTURE_WINDOW_LOST")
                        time.sleep(0.5)
                        continue
                    mon = sct.monitors[1] if region == "monitor" else region
                    shot = sct.grab(mon)
                    img = np.ascontiguousarray(np.asarray(shot)[:, :, :3])
                    scale = float(self.settings["capture_scale"])
                    if abs(scale - 1.0) > 0.01:
                        img = cv2.resize(img, None, fx=scale, fy=scale,
                                         interpolation=cv2.INTER_AREA)
                except Exception as exc:
                    self.ok = False
                    self.stats["failures"] += 1
                    self._warn_throttled(f"Capture failure: {exc}", "CAPTURE_FAIL")
                    time.sleep(0.25)
                    continue

                # health: black-frame detection (protected / exclusive fullscreen)
                if img[::16, ::16].mean() < 1.0:
                    black_run += 1
                    self.stats["black_frames"] += 1
                    if black_run == 45:
                        self.log.warning("capture",
                                         "Captured frames are black - the game may use "
                                         "exclusive fullscreen or protected content. "
                                         "Try borderless-window mode.",
                                         event_id="CAPTURE_BLACK")
                else:
                    black_run = 0
                self.ok = True

                index += 1
                frame = Frame(index, time.time(), img)
                with self._lock:
                    self._latest = frame
                self.stats["frames"] = index
                self.stats["size"] = (img.shape[1], img.shape[0])
                self._record(frame, fps)
                for cb in list(self.subscribers):
                    try:
                        cb(frame)
                    except Exception as exc:
                        self.log.error("capture", f"Subscriber error: {exc}")

                fps_n += 1
                now = time.perf_counter()
                if now - fps_t0 >= 1.0:
                    self.stats["fps"] = fps_n / (now - fps_t0)
                    fps_t0, fps_n = now, 0
                elapsed = time.perf_counter() - t0
                if elapsed > interval * 1.5:
                    self.stats["drops"] += int(elapsed / interval) - 1
                time.sleep(max(0.0, interval - elapsed))
