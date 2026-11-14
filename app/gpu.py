"""GPU helpers (AMD-friendly, no CUDA needed).

What runs on the GPU:
  * Screen capture  - DXGI Desktop Duplication through `dxcam` (any GPU, incl. AMD).
  * Image processing - OpenCV T-API/OpenCL (`cv2.UMat`) for template matching, resize, colour conversion.
  * Object detection - YOLO exported to ONNX, run by ONNX Runtime + DirectML (works on AMD/Intel/NVIDIA).
  * Video recording  - AMD hardware encoder `h264_amf` through ffmpeg.
What stays on the CPU: Python control flow, Tesseract OCR, the kNN policy, database/logging.
"""
from __future__ import annotations

import shutil
import subprocess

import cv2


class _Accel:
    """OpenCL switch used by the perception code (cv2.UMat)."""

    def __init__(self):
        self.enabled = False
        self.device = ""

    def configure(self, want: bool, log=None):
        try:
            have = bool(cv2.ocl.haveOpenCL())
            cv2.ocl.setUseOpenCL(bool(want) and have)
            self.enabled = bool(want) and have and bool(cv2.ocl.useOpenCL())
            self.device = cv2.ocl.Device.getDefault().name() if have else ""
        except Exception:
            self.enabled, self.device = False, ""
        if log:
            if self.enabled:
                log.info("gpu", f"OpenCL acceleration ON ({self.device})", event_id="GPU_OPENCL_ON")
            elif want:
                log.warning("gpu", "OpenCL not available - image processing stays on the CPU. "
                            "Install the latest AMD Adrenalin driver (it provides OpenCL).",
                            event_id="GPU_OPENCL_OFF")
        return self.enabled

    def disable(self, log=None, why: str = ""):
        if self.enabled:
            self.enabled = False
            try:
                cv2.ocl.setUseOpenCL(False)
            except Exception:
                pass
            if log:
                log.warning("gpu", f"OpenCL disabled after an error: {why}", event_id="GPU_OPENCL_ERR")

    def up(self, a):
        return cv2.UMat(a) if self.enabled else a

    @staticmethod
    def down(a):
        return a.get() if isinstance(a, cv2.UMat) else a


accel = _Accel()

_AMF_CACHE: dict = {}


def find_ffmpeg(settings) -> str | None:
    custom = str(settings.get("ffmpeg_path", "") or "").strip()
    return custom if custom else shutil.which("ffmpeg")


def ffmpeg_has_amf(path: str | None) -> bool:
    if not path:
        return False
    if path not in _AMF_CACHE:
        try:
            out = subprocess.run([path, "-hide_banner", "-encoders"], capture_output=True,
                                 text=True, timeout=10).stdout
            _AMF_CACHE[path] = "h264_amf" in out
        except Exception:
            _AMF_CACHE[path] = False
    return _AMF_CACHE[path]


def gpu_report(settings) -> dict:
    rep: dict = {"opencl": {"available": False, "enabled": accel.enabled, "device": accel.device}}
    try:
        rep["opencl"]["available"] = bool(cv2.ocl.haveOpenCL())
    except Exception:
        pass
    try:
        import dxcam
        info = {"available": True}
        try:
            info["devices"] = str(dxcam.device_info()).strip()
            info["outputs"] = str(dxcam.output_info()).strip()
        except Exception:
            pass
        rep["dxcam"] = info
    except Exception as exc:
        rep["dxcam"] = {"available": False, "error": exc.__class__.__name__}
    try:
        import onnxruntime as ort
        prov = ort.get_available_providers()
        rep["onnx"] = {"available": True, "providers": prov, "directml": "DmlExecutionProvider" in prov}
    except Exception:
        rep["onnx"] = {"available": False, "providers": [], "directml": False}
    ff = find_ffmpeg(settings)
    rep["ffmpeg"] = {"path": ff or "", "found": bool(ff), "amf": ffmpeg_has_amf(ff)}
    return rep


def report_text(rep: dict, capture_backend: str = "") -> str:
    ok = lambda b: "✔" if b else "✘"
    o, d, x, f = rep["opencl"], rep["dxcam"], rep["onnx"], rep["ffmpeg"]
    lines = []
    if capture_backend:
        lines.append(f"Active capture backend: {capture_backend}")
    lines += [
        f"{ok(d['available'])} GPU screen capture (dxcam / DXGI)"
        + ("" if d["available"] else "  → pip install dxcam"),
        f"{ok(o['enabled'])} OpenCL image processing"
        + (f"  ({o['device']})" if o["enabled"] else "  → update the AMD Adrenalin driver"
           if not o["available"] else "  (switched off in Settings or failed at runtime)"),
        f"{ok(x['directml'])} YOLO on GPU (ONNX Runtime + DirectML)"
        + ("" if x["directml"] else "  → pip install onnxruntime-directml"),
        f"{ok(f['amf'])} Hardware video encoding (ffmpeg h264_amf)"
        + ("" if f["amf"] else "  → install a full ffmpeg build and put it on PATH"
           if not f["found"] else "  → this ffmpeg has no AMF encoder"),
        "", "Still on the CPU: Python control flow, Tesseract OCR, kNN policy, database, logging.",
        "Frames must be copied to system memory for Python/OpenCV, so some CPU use remains.",
    ]
    if d.get("outputs"):
        lines += ["", "Displays seen by DXGI:", d["outputs"]]
    return "\n".join(lines)
