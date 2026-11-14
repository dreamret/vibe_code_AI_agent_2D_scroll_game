"""Auto-labelling: find candidate game elements, group look-alikes, suggest labels.

This is *assisted* labelling, not magic: regions are proposed from edges and
motion, similar-looking regions are clustered, and each cluster gets a suggested
label (nearest known template, otherwise a generic hint such as `ui` or
`moving_entity`). Nothing is stored as a label until the human applies it.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .imgio import imread

HINT_LABELS = {"ui": "ui", "moving_entity": "moving_entity",
               "item_or_marker": "item", "terrain": "terrain"}
HINT_CATEGORY = {"ui": "ui", "moving_entity": "enemy", "item_or_marker": "item",
                 "terrain": "terrain"}
MATCH_THRESHOLD = 0.80


def feature(crop: np.ndarray) -> np.ndarray:
    """Small appearance descriptor: colour histogram + normalised gray thumbnail."""
    g = cv2.resize(crop, (24, 24), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(g, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-6)
    thumb = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY).astype(np.float32).flatten() / 255.0
    thumb -= thumb.mean()
    thumb /= (np.linalg.norm(thumb) + 1e-6)
    v = np.concatenate([hist * 3.0, thumb]).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-6)


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix, iy = max(ax, bx), max(ay, by)
    ox, oy = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, ox - ix) * max(0, oy - iy)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def guess_hint(box, shape, moving: bool) -> str:
    x, y, bw, bh = box
    h, w = shape[:2]
    if (bw > 0.4 * w and bh < 0.12 * h) or (bh > 0.4 * h and bw < 0.12 * w):
        return "ui"                                   # bars / panels
    if not moving and bh < 0.15 * h and (y < 0.15 * h or y + bh > 0.85 * h):
        return "ui"                                   # HUD strip
    if moving and 0.02 * h <= bh <= 0.4 * h:
        return "moving_entity"                        # character / enemy / NPC / effect
    if not moving and bw > 0.25 * w and y > 0.6 * h:
        return "terrain"
    return "item_or_marker" if bw < 0.12 * w and bh < 0.12 * h else "terrain"


def propose_regions(img: np.ndarray, prev: np.ndarray | None = None, existing=(),
                    min_size: int = 14, max_frac: float = 0.25, max_regions: int = 40) -> list[dict]:
    """Candidate element boxes from edges (+ motion against the previous frame)."""
    h, w = img.shape[:2]
    s = 640.0 / w if w > 640 else 1.0
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s != 1.0 else img
    sh, sw = small.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    k = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(cv2.Canny(gray, 50, 140), k, iterations=2)
    mot = None
    if prev is not None:
        pg = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(prev, (sw, sh)), cv2.COLOR_BGR2GRAY), (3, 3), 0)
        mot = cv2.dilate((cv2.absdiff(gray, pg) > 25).astype(np.uint8) * 255, k, iterations=2)
        mask = cv2.bitwise_or(mask, mot)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_s = max(6, int(min_size * s))
    out = []
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < min_s or bh < min_s or bw * bh > max_frac * sw * sh:
            continue
        if not (1 / 12 <= bw / bh <= 12):
            continue
        moving = bool(mot is not None and (mot[y:y + bh, x:x + bw] > 0).mean() > 0.10)
        bx, by = int(x / s), int(y / s)
        box = (bx, by, min(int(bw / s), w - bx), min(int(bh / s), h - by))
        if any(iou(box, e) > 0.3 for e in existing):
            continue
        out.append({"bbox": box, "moving": moving, "hint": guess_hint(box, img.shape, moving)})
    out.sort(key=lambda r: -r["bbox"][2] * r["bbox"][3])
    return out[:max_regions]


def load_library(templates_root: Path, profile_id: str) -> list[tuple]:
    """Features of already-labelled templates: [(label, name, feature)]."""
    lib = []
    root = Path(templates_root) / profile_id
    if not root.exists():
        return lib
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for f in list(d.glob("*.png")) + list(d.glob("*.jpg")):
            img = imread(f)
            if img is not None and min(img.shape[:2]) >= 4:
                lib.append((d.name, f.stem, feature(img)))
    return lib


def suggest_label(centroid: np.ndarray, hint: str, library: list[tuple]) -> tuple[str, float, bool]:
    """-> (label, confidence, matched_known_template)"""
    best, bs = None, MATCH_THRESHOLD
    for label, _name, feat in library:
        sim = float(np.dot(centroid, feat))
        if sim > bs:
            best, bs = label, sim
    if best:
        return best, bs, True
    return HINT_LABELS.get(hint, "unknown"), 0.0, False


def scan_frames(frames: list[np.ndarray], existing_per_frame: list[list] | None = None,
                library: list[tuple] | None = None, min_count: int | None = None,
                threshold: float = 0.90, progress=None, max_candidates: int = 800) -> list[dict]:
    """Propose + cluster candidates across frames. Returns clusters sorted by frequency."""
    library = library or []
    cands = []
    prev = None
    for i, img in enumerate(frames):
        existing = [e.bbox for e in (existing_per_frame[i] if existing_per_frame else [])]
        for r in propose_regions(img, prev, existing):
            x, y, bw, bh = r["bbox"]
            crop = img[y:y + bh, x:x + bw]
            if crop.size == 0:
                continue
            r.update(frame_index=i, crop=crop.copy(), feat=feature(crop))
            cands.append(r)
        prev = img
        if progress:
            progress(i + 1, len(frames))
        if len(cands) >= max_candidates:
            break
    if min_count is None:
        min_count = 2 if len(frames) > 1 else 1

    clusters: list[dict] = []
    for c in cands:
        best, bs = None, threshold
        for cl in clusters:
            sim = float(np.dot(cl["centroid"], c["feat"]))
            if sim > bs:
                best, bs = cl, sim
        if best is None:
            clusters.append({"centroid": c["feat"].copy(), "members": [c]})
        else:
            best["members"].append(c)
            n = len(best["members"])
            cen = best["centroid"] * (n - 1) / n + c["feat"] / n
            best["centroid"] = cen / (np.linalg.norm(cen) + 1e-6)

    result = []
    for cl in sorted(clusters, key=lambda c: -len(c["members"])):
        m = cl["members"]
        if len(m) < min_count:
            continue
        hint = Counter(x["hint"] for x in m).most_common(1)[0][0]
        label, conf, known = suggest_label(cl["centroid"], hint, library)
        seen, samples = set(), []
        for x in m:                                       # up to 4 samples from distinct frames
            if x["frame_index"] not in seen:
                seen.add(x["frame_index"])
                samples.append({"frame_index": x["frame_index"], "bbox": x["bbox"], "crop": x["crop"]})
            if len(samples) == 4:
                break
        result.append({
            "id": len(result), "count": len(m), "hint": hint,
            "moving_ratio": round(float(np.mean([x["moving"] for x in m])), 2),
            "size": [int(np.median([x["bbox"][2] for x in m])), int(np.median([x["bbox"][3] for x in m]))],
            "suggested_label": label, "match_confidence": round(conf, 3), "known": known,
            "category": HINT_CATEGORY.get(hint, "other"), "samples": samples,
            "all_boxes": [(x["frame_index"], x["bbox"]) for x in m],
        })
    return result
