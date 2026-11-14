"""Image read/write that also works with non-ASCII (e.g. Chinese/Japanese) paths on Windows."""
import cv2
import numpy as np


def imread(path, flags=cv2.IMREAD_COLOR):
    try:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)
    except Exception:
        return None


def imwrite(path, img, params=None) -> bool:
    ext = str(path).rsplit(".", 1)[-1] if "." in str(path) else "png"
    ok, buf = cv2.imencode("." + ext, img, params or [])
    if ok:
        buf.tofile(str(path))
    return bool(ok)
