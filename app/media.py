"""Label + analyse images, GIFs and short videos."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import autolabel
from .db import now_iso
from .perception import (Tracker, detect_captcha, detect_text_language, extract_scene,
                         infer_class, Entity)

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv"}
MEDIA_FILTER = ("Media (*.gif *.mp4 *.avi *.mov *.mkv *.webm *.m4v *.wmv "
                "*.png *.jpg *.jpeg *.bmp *.webp);;All files (*.*)")


@dataclass
class MediaResult:
    info: dict
    frames: list = field(default_factory=list)          # BGR images (sampled)
    times: list = field(default_factory=list)           # seconds
    entities: list = field(default_factory=list)        # list[list[Entity]]
    texts: list = field(default_factory=list)           # list[list[str]]
    summary: dict = field(default_factory=dict)
    clusters: list = field(default_factory=list)

    def to_doc(self, profile_id: str) -> dict:
        return {"profile_id": profile_id, "path": self.info["path"], "type": self.info["type"],
                "info": self.info, "summary": self.summary, "created": now_iso(),
                "frames": [{"index": i, "t": round(t, 3),
                            "entities": [e.to_dict() for e in ents], "text": txt[:20]}
                           for i, (t, ents, txt) in enumerate(zip(self.times, self.entities, self.texts))],
                "clusters": [{k: c[k] for k in ("id", "count", "hint", "suggested_label",
                                                "match_confidence", "size", "known")}
                             for c in self.clusters]}


def _read_image(path: str):
    data = np.fromfile(path, dtype=np.uint8)              # works with non-ASCII paths
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def load_media(path: str, max_frames: int = 48, max_width: int = 1280, max_seconds: float = 30.0):
    p = Path(path)
    ext = p.suffix.lower()
    info = {"path": str(p), "name": p.name, "scale": 1.0, "truncated": False}
    frames, times = [], []
    if ext in IMAGE_EXT:
        img = _read_image(str(p))
        if img is None:
            raise ValueError("Cannot read image")
        info.update(type="image", fps=0.0, duration=0.0, total_frames=1)
        frames, times = [img], [0.0]
    elif ext == ".gif":
        from PIL import Image
        im = Image.open(str(p))
        n = getattr(im, "n_frames", 1)
        durs = []
        for i in range(n):
            im.seek(i)
            durs.append(max(10, int(im.info.get("duration", 100))) / 1000.0)
        cum = np.concatenate([[0.0], np.cumsum(durs)])
        end = n
        if cum[-1] > max_seconds:
            end = int(np.searchsorted(cum, max_seconds))
            info["truncated"] = True
        idx = np.unique(np.linspace(0, max(0, end - 1), min(max_frames, end)).astype(int))
        for i in idx:
            im.seek(int(i))
            frames.append(cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR))
            times.append(float(cum[i]))
        info.update(type="gif", fps=round(n / cum[-1], 2) if cum[-1] else 0.0,
                    duration=round(float(cum[-1]), 2), total_frames=n)
    elif ext in VIDEO_EXT:
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            raise ValueError("Cannot open video")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        end = min(total, int(max_seconds * fps)) if total else int(max_seconds * fps)
        info["truncated"] = bool(total and total > end)
        idx = np.unique(np.linspace(0, max(0, end - 1), min(max_frames, max(1, end))).astype(int))
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if ok:
                frames.append(f)
                times.append(float(i / fps))
        cap.release()
        info.update(type="video", fps=round(fps, 2), duration=round(total / fps, 2) if total else 0.0,
                    total_frames=total)
    else:
        raise ValueError(f"Unsupported file type '{ext}'")
    if not frames:
        raise ValueError("No frames could be read")
    h, w = frames[0].shape[:2]
    info["orig_size"] = [w, h]
    if w > max_width:
        s = max_width / w
        frames = [cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) for f in frames]
        info["scale"] = round(s, 4)
    info["size"] = [frames[0].shape[1], frames[0].shape[0]]
    return frames, times, info


def analyze_media(path: str, perception, library, progress=None, stop=None,
                  max_frames: int = 48) -> MediaResult:
    """Detect known elements, read text (OCR), flag CAPTCHAs, infer class, and
    propose labels for unknown recurring elements."""
    frames, times, info = load_media(path, max_frames=max_frames)
    res = MediaResult(info=info, frames=frames, times=times)
    profile = perception.profile
    labels = profile["detector"]["labels"]
    td, yolo, ocr = perception.template_det, perception.yolo, perception.ocr
    tracker = Tracker()
    text_now: list[str] = []
    per_label_frames = defaultdict(int)
    per_label_max = defaultdict(int)
    per_label_tracks = defaultdict(set)
    all_text = Counter()
    captcha_hits, class_votes = [], Counter()
    diffs, changes = [], 0
    prev_small = None
    n = len(frames)
    for i, img in enumerate(frames):
        if stop is not None and stop.is_set():
            break
        ents: list[Entity] = []
        if td:
            ents += td.detect(img)
        if yolo:
            try:
                ents += yolo.detect(img)
            except Exception:
                pass
        tracker.update(ents, img.shape[1])
        if ocr and i % 4 == 0:
            try:
                text_now = ocr.read(img)
            except Exception:
                text_now = []
        cap_ents = td.detect(img, only="_captcha") if td and "_captcha" in td.templates else []
        cap = detect_captcha(text_now, cap_ents, profile["detector"].get("captcha_keywords", []))
        if cap.get("detected"):
            captcha_hits.append((i, cap.get("reason", "")))
        class_votes[infer_class(profile, ents, text_now)["candidate"]] += 1
        for line in text_now:
            all_text[line] += 1
        counts = Counter(e.label for e in ents)
        for lab, c in counts.items():
            per_label_frames[lab] += 1
            per_label_max[lab] = max(per_label_max[lab], c)
        for e in ents:
            per_label_tracks[e.label].add(e.track_id)
        small = cv2.cvtColor(cv2.resize(img, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev_small is not None:
            d = float(np.abs(small - prev_small).mean())
            diffs.append(d)
            changes += d > 30
        prev_small = small
        res.entities.append(ents)
        res.texts.append(list(text_now))
        if progress:
            progress(i + 1, n)

    # auto-label unknown recurring elements
    res.clusters = autolabel.scan_frames(frames, res.entities, library)

    lines = [t for t, _ in all_text.most_common(30)]
    lang = detect_text_language(lines)
    motion = float(np.mean(diffs)) if diffs else 0.0
    cls = class_votes.most_common(1)[0][0] if class_votes else "Unknown"
    notes = []
    if info.get("truncated"):
        notes.append("Media is longer than 30 s: only the first 30 s were analysed.")
    if info.get("scale", 1.0) != 1.0:
        notes.append(f"Frames were scaled by {info['scale']} for memory; templates saved from this "
                     "media only match a game window of the same size.")
    if not td or not td.templates:
        notes.append("No templates for this profile yet - only auto-label suggestions are available.")
    if not ocr:
        notes.append("OCR is off/unavailable - no text was read.")
    res.summary = {
        "frames_analyzed": len(res.entities), "duration_sec": info.get("duration", 0.0),
        "size": info["size"],
        "labels": {l: {"frames_present": per_label_frames[l], "max_per_frame": per_label_max[l],
                       "tracks": len(per_label_tracks[l])} for l in per_label_frames},
        "text": {"lines": lines, "language": lang["language"], "script_counts": lang["counts"]},
        "captcha": {"frames": len(captcha_hits), "reason": captcha_hits[0][1] if captcha_hits else ""},
        "class_inference": cls,
        "motion": {"mean_change": round(motion, 2),
                   "level": "static" if motion < 1 else "low" if motion < 6 else
                   "medium" if motion < 15 else "high"},
        "scene_changes": int(changes), "unlabeled_clusters": len(res.clusters), "notes": notes,
    }
    return res


def summary_text(res: MediaResult) -> str:
    s, i = res.summary, res.info
    out = [f"{i['name']}  ({i['type']}, {i['size'][0]}x{i['size'][1]}, "
           f"{s['frames_analyzed']} frames analysed, {s['duration_sec']} s)"]
    if s["labels"]:
        out.append("Known elements: " + ", ".join(
            f"{l} (max {v['max_per_frame']}/frame, {v['tracks']} track(s))" for l, v in s["labels"].items()))
    else:
        out.append("Known elements: none detected")
    out.append(f"Motion: {s['motion']['level']}   scene changes: {s['scene_changes']}")
    out.append(f"Text language: {s['text']['language']}")
    if s["text"]["lines"]:
        out.append("Text: " + " | ".join(s["text"]["lines"][:8]))
    out.append(f"Character/class: {s['class_inference']}")
    if s["captcha"]["frames"]:
        out.append(f"CAPTCHA-like content in {s['captcha']['frames']} frame(s): {s['captcha']['reason']}")
    out.append(f"Unlabelled recurring elements found: {s['unlabeled_clusters']} (see suggestions)")
    out += [f"Note: {n}" for n in s["notes"]]
    return "\n".join(out)


# ----------------------------------------------------------------- export
def _draw_text(img, text, xy, color):
    if text.isascii():
        cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return img
    try:                                                   # CJK labels via Pillow
        from PIL import Image, ImageDraw, ImageFont
        font = None
        for name in ("msjh.ttc", "meiryo.ttc", "msgothic.ttc", "NotoSansCJK-Regular.ttc"):
            try:
                font = ImageFont.truetype(name, 16)
                break
            except Exception:
                continue
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ImageDraw.Draw(pil).text((xy[0], xy[1] - 16), text, fill=color[::-1], font=font)
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(img, "?", xy, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return img


def annotate_frame(img, ents):
    from .gui_colors import bgr_color
    out = img.copy()
    for e in ents:
        x, y, w, h = e.bbox
        col = bgr_color(e.label)
        cv2.rectangle(out, (x, y), (x + w, y + h), col, 2)
        out = _draw_text(out, f"{e.label} {e.confidence:.2f}", (x, max(14, y - 4)), col)
    return out


def export_annotated(res: MediaResult, path: str):
    ann = [annotate_frame(f, e) for f, e in zip(res.frames, res.entities)]
    if path.lower().endswith(".gif"):
        from PIL import Image
        ims = [Image.fromarray(cv2.cvtColor(a, cv2.COLOR_BGR2RGB)) for a in ann]
        dur = int(1000 * max(0.05, (res.times[1] - res.times[0]) if len(res.times) > 1 else 0.2))
        ims[0].save(path, save_all=True, append_images=ims[1:], duration=dur, loop=0)
    elif res.info["type"] == "image":
        cv2.imencode(Path(path).suffix or ".png", ann[0])[1].tofile(path)
    else:
        step = (res.times[-1] - res.times[0]) / max(1, len(res.times) - 1) if len(res.times) > 1 else 0.2
        h, w = ann[0].shape[:2]
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, 1 / max(step, 0.04)), (w, h))
        for a in ann:
            vw.write(a)
        vw.release()


def export_report(res: MediaResult, profile_id: str, path: str):
    Path(path).write_text(json.dumps(res.to_doc(profile_id), indent=1, ensure_ascii=False,
                                     default=str), encoding="utf-8")
