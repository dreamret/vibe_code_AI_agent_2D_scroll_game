"""Perception service: detection, tracking, OCR, CAPTCHA/marker + class inference."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

BUILTIN_CAPTCHA_KEYWORDS = [
    "captcha", "recaptcha", "hcaptcha", "i'm not a robot", "i am not a robot",
    "verify you are human", "verify you're human", "are you human",
    "security check", "select all images", "slide to verify", "drag the slider",
    "anti-bot", "human verification",
]


@dataclass
class Entity:
    label: str
    bbox: tuple                 # x, y, w, h in frame pixels
    confidence: float
    name: str = ""              # template name / model class name
    track_id: int = -1
    source: str = "template"

    @property
    def center(self):
        x, y, w, h = self.bbox
        return (x + w / 2.0, y + h / 2.0)

    def to_dict(self):
        return {"label": self.label, "bbox": [int(v) for v in self.bbox],
                "confidence": round(float(self.confidence), 4), "name": self.name,
                "track_id": self.track_id, "source": self.source}


@dataclass
class PerceptionResult:
    frame_index: int
    timestamp: float
    image: np.ndarray
    entities: list = field(default_factory=list)
    scene: dict = field(default_factory=dict)
    captcha: dict = field(default_factory=lambda: {"detected": False})
    class_info: dict = field(default_factory=lambda: {"candidate": "Unknown",
                                                      "confidence": 0.0, "evidence": []})
    ocr_text: list = field(default_factory=list)
    proc_ms: float = 0.0


# --------------------------------------------------------------------------
class TemplateDetector:
    """Template matching with per-profile detector packs.

    Layout: data/templates/<profile_id>/<label>/*.png
            data/templates/<profile_id>/_captcha/*.png   (CAPTCHA screens/widgets)
    """

    def __init__(self, root: Path, profile_id: str, threshold: float, scale: float, log):
        self.dir = Path(root) / profile_id
        self.threshold = threshold
        self.scale = max(0.1, min(1.0, scale))
        self.log = log
        self.templates: dict[str, list] = {}
        self.reload()

    def reload(self):
        self.templates = {}
        if not self.dir.exists():
            return
        count = 0
        for label_dir in sorted(p for p in self.dir.iterdir() if p.is_dir()):
            items = []
            for f in sorted(label_dir.glob("*.png")) + sorted(label_dir.glob("*.jpg")):
                img = cv2.imread(str(f), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if self.scale != 1.0:
                    g = cv2.resize(g, None, fx=self.scale, fy=self.scale,
                                   interpolation=cv2.INTER_AREA)
                if g.shape[0] >= 4 and g.shape[1] >= 4:
                    items.append((f.stem, g))
                    count += 1
            if items:
                self.templates[label_dir.name] = items
        self.log.info("perception", f"Loaded {count} templates", event_id="TEMPLATES_LOADED",
                      labels=list(self.templates))

    def detect(self, frame: np.ndarray, only: str | None = None,
               skip_special: bool = True) -> list[Entity]:
        if not self.templates:
            return []
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.scale != 1.0:
            g = cv2.resize(g, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        out: list[Entity] = []
        for label, items in self.templates.items():
            if only is not None and label != only:
                continue
            if only is None and skip_special and label.startswith("_"):
                continue
            boxes, scores, names = [], [], []
            for name, tpl in items:
                th, tw = tpl.shape[:2]
                if th >= g.shape[0] or tw >= g.shape[1]:
                    continue
                res = cv2.matchTemplate(g, tpl, cv2.TM_CCOEFF_NORMED)
                ys, xs = np.where(res >= self.threshold)
                if len(xs) == 0:
                    continue
                order = np.argsort(res[ys, xs])[::-1][:100]
                for i in order:
                    boxes.append([int(xs[i]), int(ys[i]), tw, th])
                    scores.append(float(res[ys[i], xs[i]]))
                    names.append(name)
            if not boxes:
                continue
            keep = cv2.dnn.NMSBoxes(boxes, scores, self.threshold, 0.3)
            for i in np.array(keep).flatten():
                x, y, w, h = boxes[int(i)]
                inv = 1.0 / self.scale
                out.append(Entity(label, (int(x * inv), int(y * inv), int(w * inv), int(h * inv)),
                                  scores[int(i)], names[int(i)], source="template"))
        return out


class YoloDetector:
    """Optional custom detector (YOLO family) - requires `pip install ultralytics`."""

    def __init__(self, model_path: str, conf: float, log):
        from ultralytics import YOLO          # noqa: heavy optional dependency
        self.model = YOLO(model_path)
        self.conf = conf
        log.info("perception", f"YOLO model loaded: {model_path}", event_id="YOLO_LOADED")

    def detect(self, frame: np.ndarray) -> list[Entity]:
        r = self.model.predict(frame, conf=self.conf, verbose=False)[0]
        out = []
        for b in r.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            name = str(r.names[int(b.cls[0])])
            out.append(Entity(name, (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                              float(b.conf[0]), name, source="yolo"))
        return out


class OcrEngine:
    def __init__(self, tesseract_cmd: str, log):
        import pytesseract
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.pt = pytesseract
        self.log = log
        self.pt.get_tesseract_version()       # raises if the executable is missing

    def read(self, frame: np.ndarray) -> list[str]:
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if g.shape[1] > 1280:
            f = 1280 / g.shape[1]
            g = cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        text = self.pt.image_to_string(g)
        return [ln.strip() for ln in text.splitlines() if len(ln.strip()) >= 3][:60]


# --------------------------------------------------------------------------
class Tracker:
    """Greedy nearest-centre tracker, per label."""

    def __init__(self, max_missing=10):
        self.tracks: dict[int, dict] = {}
        self.next_id = 1
        self.max_missing = max_missing

    def update(self, entities: list[Entity], frame_w: int):
        gate = 0.12 * frame_w
        used = set()
        for e in sorted(entities, key=lambda e: -e.confidence):
            best, best_d = None, gate
            for tid, t in self.tracks.items():
                if t["label"] != e.label or tid in used:
                    continue
                d = np.hypot(t["c"][0] - e.center[0], t["c"][1] - e.center[1])
                if d < best_d:
                    best, best_d = tid, d
            if best is None:
                best = self.next_id
                self.next_id += 1
            used.add(best)
            e.track_id = best
            self.tracks[best] = {"label": e.label, "c": e.center, "missing": 0}
        for tid in list(self.tracks):
            if tid not in used:
                self.tracks[tid]["missing"] += 1
                if self.tracks[tid]["missing"] > self.max_missing:
                    del self.tracks[tid]


def extract_scene(entities: list[Entity], shape, ocr_text: list[str], labels: list[str]) -> dict:
    h, w = shape[:2]
    counts = {l: 0 for l in labels}
    pos: dict[str, list] = {}
    for e in entities:
        counts[e.label] = counts.get(e.label, 0) + 1
        pos.setdefault(e.label, []).append(e.center)
    player = None
    players = [e for e in entities if e.label == "player"]
    if players:
        p = max(players, key=lambda e: e.confidence)
        player = {"cx": p.center[0] / w, "cy": p.center[1] / h}
    nearest = None
    if player:
        best = None
        for e in entities:
            if e.label == "enemy":
                dx, dy = e.center[0] / w - player["cx"], e.center[1] / h - player["cy"]
                d = float(np.hypot(dx, dy))
                if best is None or d < best[0]:
                    best = (d, dx, dy)
        if best:
            nearest = {"dist": best[0], "dx": best[1], "dy": best[2]}
    mean_pos = {l: [float(np.mean([c[0] for c in v]) / w), float(np.mean([c[1] for c in v]) / h)]
                for l, v in pos.items()}
    return {"size": [w, h], "counts": counts, "mean_pos": mean_pos, "player": player,
            "nearest_enemy": nearest, "ocr_text": ocr_text[:15],
            "labels_present": sorted(l for l, c in counts.items() if c),
            "scene_type": "gameplay" if (player or entities) else "unknown"}


def detect_captcha(ocr_text: list[str], captcha_entities: list[Entity],
                   extra_keywords: list[str]) -> dict:
    """Detection ONLY. The application never solves or bypasses CAPTCHAs."""
    kws = BUILTIN_CAPTCHA_KEYWORDS + [k.lower() for k in extra_keywords]
    blob = " ".join(ocr_text).lower()
    hits = [k for k in kws if k in blob]
    if captcha_entities:
        best = max(captcha_entities, key=lambda e: e.confidence)
        return {"detected": True, "reason": f"CAPTCHA template '{best.name}' matched",
                "confidence": best.confidence}
    if hits:
        return {"detected": True, "reason": f"OCR keyword(s): {', '.join(hits[:3])}",
                "confidence": min(1.0, 0.6 + 0.2 * len(hits))}
    return {"detected": False}


def infer_class(profile: dict, entities: list[Entity], ocr_text: list[str]) -> dict:
    """Score each configured class by how many of its signature keywords are visible.
    Returns 'Unknown' when evidence is weak or ambiguous - never guesses."""
    sigs = profile.get("character", {}).get("class_signatures", {}) or {}
    if not sigs:
        hint = profile.get("character", {}).get("class_hint", "")
        if hint:
            return {"candidate": hint, "confidence": 0.3,
                    "evidence": ["profile class_hint (not verified visually)"]}
        return {"candidate": "Unknown", "confidence": 0.0, "evidence": []}
    seen = {e.name.lower() for e in entities} | {e.label.lower() for e in entities}
    blob = " ".join(ocr_text).lower()
    scored = []
    for cls, kws in sigs.items():
        if not kws:
            continue
        ev = [k for k in kws if k.lower() in seen or k.lower() in blob]
        scored.append((len(ev) / len(kws), cls, ev))
    scored.sort(reverse=True)
    if not scored or scored[0][0] < 0.5:
        return {"candidate": "Unknown", "confidence": scored[0][0] if scored else 0.0,
                "evidence": []}
    if len(scored) > 1 and scored[0][0] - scored[1][0] < 0.15:
        return {"candidate": "Unknown (ambiguous)", "confidence": scored[0][0],
                "evidence": [f"{scored[0][1]} vs {scored[1][1]}"]}
    return {"candidate": scored[0][1], "confidence": scored[0][0], "evidence": scored[0][2]}


# --------------------------------------------------------------------------
class PerceptionService:
    def __init__(self, settings, log):
        self.settings = settings
        self.log = log
        self.profile: dict = {}
        self.template_det: TemplateDetector | None = None
        self.yolo: YoloDetector | None = None
        self.ocr: OcrEngine | None = None
        self.tracker = Tracker()
        self.subscribers: list = []
        self._cond = threading.Condition()
        self._pending = None
        self._latest: PerceptionResult | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_ocr_t = 0.0
        self._last_ocr: list[str] = []
        self.avg_ms = 0.0

    def subscribe(self, cb):
        self.subscribers.append(cb)

    def latest(self) -> PerceptionResult | None:
        return self._latest

    def configure(self, profile: dict):
        self.profile = profile
        d = profile["detector"]
        self.tracker = Tracker()
        self.template_det = TemplateDetector(self.settings.path("templates_dir"), profile["_id"],
                                             d["template_threshold"], d["detect_scale"], self.log)
        self.yolo = None
        if d.get("yolo_model"):
            try:
                self.yolo = YoloDetector(d["yolo_model"], d["yolo_confidence"], self.log)
            except Exception as exc:
                self.log.error("perception", f"YOLO unavailable: {exc}", event_id="YOLO_FAIL")
        self.ocr = None
        if self.settings["ocr_enabled"] or d.get("use_ocr"):
            try:
                self.ocr = OcrEngine(self.settings["tesseract_cmd"], self.log)
            except Exception as exc:
                self.log.warning("perception",
                                 f"OCR disabled ({exc.__class__.__name__}): install Tesseract "
                                 "and `pip install pytesseract` for text/CAPTCHA keyword detection",
                                 event_id="OCR_UNAVAILABLE")

    def reload_templates(self):
        if self.template_det:
            self.template_det.reload()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="perception")
        self._thread.start()

    def stop(self):
        self._stop.set()
        with self._cond:
            self._cond.notify_all()

    def submit(self, frame):
        with self._cond:
            self._pending = frame          # keep only the newest frame (drop old)
            self._cond.notify()

    def _loop(self):
        while not self._stop.is_set():
            with self._cond:
                while self._pending is None and not self._stop.is_set():
                    self._cond.wait(0.5)
                frame, self._pending = self._pending, None
            if frame is None:
                continue
            try:
                res = self._process(frame)
            except Exception as exc:
                self.log.error("perception", f"Processing error: {exc}", event_id="PERCEPTION_ERR")
                continue
            self._latest = res
            for cb in list(self.subscribers):
                try:
                    cb(res)
                except Exception as exc:
                    self.log.error("perception", f"Subscriber error: {exc}")

    def _process(self, frame) -> PerceptionResult:
        t0 = time.perf_counter()
        img = frame.image
        d = self.profile["detector"]
        entities: list[Entity] = []
        if self.template_det:
            entities += self.template_det.detect(img)
        if self.yolo:
            entities += self.yolo.detect(img)
        self.tracker.update(entities, img.shape[1])

        now = time.time()
        if self.ocr and now - self._last_ocr_t >= float(self.settings["ocr_interval_sec"]):
            self._last_ocr_t = now
            try:
                self._last_ocr = self.ocr.read(img)
            except Exception as exc:
                self.log.warning("perception", f"OCR failed: {exc}", event_id="OCR_FAIL")
                self.ocr = None
        ocr_text = self._last_ocr if self.ocr else []

        cap_ents = []
        if self.template_det and "_captcha" in self.template_det.templates:
            cap_ents = self.template_det.detect(img, only="_captcha")
        captcha = detect_captcha(ocr_text, cap_ents, d.get("captcha_keywords", []))
        scene = extract_scene(entities, img.shape, ocr_text, d["labels"])
        cls = infer_class(self.profile, entities, ocr_text)
        ms = (time.perf_counter() - t0) * 1000
        self.avg_ms = 0.9 * self.avg_ms + 0.1 * ms if self.avg_ms else ms
        return PerceptionResult(frame.index, frame.timestamp, img, entities, scene,
                                captcha, cls, ocr_text, ms)
