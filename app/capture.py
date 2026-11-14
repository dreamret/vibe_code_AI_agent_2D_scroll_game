"""Capture service: GPU (DXGI) or CPU (mss) capture, recording, health checks."""
from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import gpu, winutil


@dataclass
class Frame:
    index: int
    timestamp: float
    image: np.ndarray            # BGR


# --------------------------------------------------------------------- backends
class _MssBackend:
    name = "mss (CPU / GDI)"

    def __init__(self):
        import mss
        self._ctx = mss.mss()
        self.monitor = self._ctx.monitors[1]

    def grab(self, region):
        shot = self._ctx.grab(self.monitor if region == "monitor" else region)
        return np.ascontiguousarray(np.asarray(shot)[:, :, :3])

    def close(self):
        try:
            self._ctx.close()
        except Exception:
            pass


class _DxcamBackend:
    """DXGI Desktop Duplication: the GPU/driver does the capture work (works on AMD).
    Captures a screen region, so the game window must be visible and not covered."""
    name = "dxcam (GPU / DXGI)"
    _cams: dict = {}                      # kept for the process lifetime (dxcam keeps D3D resources)

    def __init__(self, device_idx: int = 0):
        import dxcam
        import mss
        self.dxcam = dxcam
        self.dev = device_idx
        with mss.mss() as s:
            self.mons = [dict(m) for m in s.monitors[1:]]
        self._cam(0)                      # fail early if DXGI duplication is unavailable
        self._last: dict = {}
        self._seed = None

    def _cam(self, out_idx: int):
        key = (self.dev, out_idx)
        if key not in _DxcamBackend._cams:
            _DxcamBackend._cams[key] = self.dxcam.create(device_idx=self.dev, output_idx=out_idx,
                                                          output_color="BGR")
        return _DxcamBackend._cams[key]

    def _monitor_for(self, r) -> int:
        cx, cy = r["left"] + r["width"] // 2, r["top"] + r["height"] // 2
        for i, m in enumerate(self.mons):
            if m["left"] <= cx < m["left"] + m["width"] and m["top"] <= cy < m["top"] + m["height"]:
                return i
        return 0

    def grab(self, region):
        if region == "monitor":
            m = self.mons[0]
            region = {"left": m["left"], "top": m["top"], "width": m["width"], "height": m["height"]}
        i = self._monitor_for(region)
        m = self.mons[i]
        l = max(0, region["left"] - m["left"])
        t = max(0, region["top"] - m["top"])
        r = min(m["width"], region["left"] - m["left"] + region["width"])
        b = min(m["height"], region["top"] - m["top"] + region["height"])
        if r - l < 16 or b - t < 16:
            return None
        key = (i, l, t, r, b)
        frame = self._cam(i).grab(region=(l, t, r, b))
        if frame is None:                 # DXGI only delivers when the screen changed
            if key in self._last:
                return self._last[key]
            if self._seed is None:        # very first frame of a static screen
                self._seed = _MssBackend()
            frame = self._seed.grab({"left": region["left"], "top": region["top"],
                                     "width": r - l, "height": b - t})
        frame = np.ascontiguousarray(frame)
        self._last[key] = frame
        return frame

    def close(self):
        if self._seed:
            self._seed.close()


# --------------------------------------------------------------------- recorder
class _FfmpegAmfWriter:
    """Hardware-encoded recording: raw frames -> ffmpeg -> h264_amf (AMD encoder block)."""

    def __init__(self, ffmpeg: str, path: Path, size, fps: float):
        w, h = size[0] - size[0] % 2, size[1] - size[1] % 2
        self.size = (w, h)
        self.log_f = open(str(path) + ".ffmpeg.log", "w")
        self.proc = subprocess.Popen(
            [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-", "-c:v", "h264_amf",
             "-quality", "speed", "-b:v", "8M", "-pix_fmt", "yuv420p", str(path)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log_f)

    def write(self, img: np.ndarray):
        w, h = self.size
        self.proc.stdin.write(memoryview(np.ascontiguousarray(img[:h, :w])))

    def alive(self) -> bool:
        return self.proc.poll() is None

    def release(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=15)
        except Exception:
            self.proc.kill()
        self.log_f.close()


class CaptureService:
    """Backend `auto`: dxcam (GPU) when installed on Windows, otherwise mss (CPU).
    The backend is isolated in `_make_backend` so another one can be added easily."""

    def __init__(self, settings, log):
        self.settings = settings
        self.log = log
        self.target: dict | None = None
        self.subscribers: list = []
        self.stats = {"frames": 0, "drops": 0, "failures": 0, "fps": 0.0,
                      "black_frames": 0, "size": (0, 0)}
        self.ok = False
        self.running = False
        self.backend_name = "-"
        self.encoder_name = "-"
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
            self._writer = None           # created lazily on the first frame (needs size)
            self._rec_size = None
        self.log.info("capture", f"Recording to {session_dir}", event_id="REC_START")

    def stop_recording(self):
        with self._rec_lock:
            if self._writer is not None:
                self._writer.release()
            if self._ts_file:
                self._ts_file.close()
            self._writer = self._ts_file = self._rec_dir = self._rec_size = None

    def _open_writer(self, w: int, h: int, fps: float):
        mode = str(self.settings.get("record_encoder", "auto"))
        if mode in ("auto", "amf"):
            ff = gpu.find_ffmpeg(self.settings)
            if gpu.ffmpeg_has_amf(ff):
                try:
                    wr = _FfmpegAmfWriter(ff, self._rec_dir / "video.mp4", (w, h), fps)
                    self._rec_size = wr.size
                    self.encoder_name = "h264_amf (GPU)"
                    self.log.info("capture", "Recording with the AMD hardware encoder (h264_amf)",
                                  event_id="REC_AMF")
                    return wr
                except Exception as exc:
                    self.log.warning("capture", f"AMF encoder failed ({exc}); using CPU encoder",
                                     event_id="REC_AMF_FAIL")
            elif mode == "amf":
                self.log.warning("capture", "record_encoder=amf but ffmpeg with h264_amf was not "
                                 "found; using the CPU encoder", event_id="REC_AMF_MISSING")
        self._rec_size = (w, h)
        self.encoder_name = "mp4v (CPU)"
        return cv2.VideoWriter(str(self._rec_dir / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                               fps, (w, h))

    def _record(self, frame: Frame, fps: float):
        with self._rec_lock:
            if self._rec_dir is None:
                return
            if self.settings["record_video"]:
                h, w = frame.image.shape[:2]
                if self._writer is None:
                    self._writer = self._open_writer(w, h, fps)
                img = frame.image
                if (w, h) != (self._rec_size[0], self._rec_size[1]) and \
                        not isinstance(self._writer, _FfmpegAmfWriter):
                    img = cv2.resize(img, self._rec_size)
                elif isinstance(self._writer, _FfmpegAmfWriter) and (w, h) != self._rec_size:
                    img = cv2.resize(img, self._rec_size)
                try:
                    self._writer.write(img)
                    if isinstance(self._writer, _FfmpegAmfWriter) and not self._writer.alive():
                        raise BrokenPipeError("ffmpeg exited")
                except Exception as exc:
                    self.log.warning("capture", f"Hardware encoder stopped ({exc}); the rest of this "
                                     "recording uses the CPU encoder (video_cpu.mp4)", event_id="REC_FALLBACK")
                    try:
                        self._writer.release()
                    except Exception:
                        pass
                    self._rec_size = (w, h)
                    self.encoder_name = "mp4v (CPU)"
                    self._writer = cv2.VideoWriter(str(self._rec_dir / "video_cpu.mp4"),
                                                   cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    self._writer.write(frame.image)
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

    def _make_backend(self):
        want = str(self.settings.get("capture_backend", "auto"))
        if want in ("auto", "dxcam") and winutil.IS_WINDOWS:
            try:
                return _DxcamBackend(int(self.settings.get("gpu_device_index", 0)))
            except Exception as exc:
                level = self.log.warning if want == "dxcam" else self.log.info
                level("capture", f"GPU capture (dxcam) unavailable ({exc.__class__.__name__}: {exc}); "
                      "using CPU capture (mss). `pip install dxcam` enables GPU capture.",
                      event_id="CAPTURE_NO_GPU")
        return _MssBackend()

    def _loop(self):
        try:
            backend = self._make_backend()
        except ImportError:
            self.log.error("capture", "Package 'mss' is not installed", event_id="CAPTURE_DEP")
            self.running = False
            return
        self.backend_name = backend.name
        self.log.info("capture", f"Capture backend: {backend.name}", event_id="CAPTURE_BACKEND")
        index = 0
        black_run = 0
        fps_t0, fps_n = time.perf_counter(), 0
        try:
            while not self._stop.is_set():
                fps = max(1, int(self.settings["capture_fps"]))
                interval = 1.0 / fps
                t0 = time.perf_counter()
                try:
                    region = self._region()
                    if region is None:
                        self.ok = False
                        self.stats["failures"] += 1
                        self._warn_throttled("Target window lost or minimised", "CAPTURE_WINDOW_LOST")
                        time.sleep(0.5)
                        continue
                    img = backend.grab(region)
                    if img is None:
                        time.sleep(0.005)
                        continue
                    scale = float(self.settings["capture_scale"])
                    if abs(scale - 1.0) > 0.01:
                        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                except Exception as exc:
                    if not isinstance(backend, _MssBackend):      # GPU backend broke -> CPU fallback
                        self.log.warning("capture", f"GPU capture failed ({exc}); switching to CPU "
                                         "capture (mss)", event_id="CAPTURE_GPU_FALLBACK")
                        backend.close()
                        backend = _MssBackend()
                        self.backend_name = backend.name
                        continue
                    self.ok = False
                    self.stats["failures"] += 1
                    self._warn_throttled(f"Capture failure: {exc}", "CAPTURE_FAIL")
                    time.sleep(0.25)
                    continue

                if img[::16, ::16].mean() < 1.0:               # black frames: protected/exclusive FS
                    black_run += 1
                    self.stats["black_frames"] += 1
                    if black_run == 45:
                        self.log.warning("capture", "Captured frames are black - the game may use "
                                         "exclusive fullscreen or protected content. Try borderless "
                                         "window mode.", event_id="CAPTURE_BLACK")
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
        finally:
            backend.close()
