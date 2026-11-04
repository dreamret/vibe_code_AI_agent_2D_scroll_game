"""Profiles: game/window association, control mapping, detector + behaviour."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .db import new_id, now_iso

DEFAULT_LABELS = ["player", "enemy", "npc", "item", "skill_effect", "ui", "marker", "terrain"]

PROFILE_TEMPLATE = {
    "name": "Default 2D Side-Scroller",
    "version": 1,
    "game": {"window_title_contains": "", "process_name": ""},
    "input_mapping": {
        "move_left": "left", "move_right": "right", "jump": "space",
        "move_up": "up", "move_down": "down",
        "attack": "z", "skill_1": "x", "skill_2": "c", "skill_3": "a",
        "interact": "e", "click": "mouse_left",
    },
    "character": {
        "class_hint": "",
        # class name -> list of keywords matched against detected template
        # names / labels / OCR text, e.g. {"Warrior": ["warrior_icon", "sword_skill"]}
        "class_signatures": {},
    },
    "detector": {
        "labels": DEFAULT_LABELS,
        "template_threshold": 0.80,
        "detect_scale": 0.5,
        "yolo_model": "",             # path to a YOLO .pt file (needs `ultralytics`)
        "yolo_confidence": 0.4,
        "use_ocr": False,
        "captcha_keywords": [],       # extra keywords, besides the built-in list
    },
    "behavior": {"preferred": "balanced", "notes": ""},
    "constraints": {
        # empty list = only controls listed in input_mapping are allowed
        "allowed_controls": [],
        "max_actions_per_second": 10,
    },
    "model_version": 0,
    "active_model_id": "",
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def validate(profile: dict) -> dict:
    """Fill defaults and check types. Raises ValueError for bad input."""
    p = _merge(PROFILE_TEMPLATE, profile)
    if not str(p.get("name", "")).strip():
        raise ValueError("Profile needs a name")
    if not isinstance(p["input_mapping"], dict):
        raise ValueError("input_mapping must be an object")
    if not isinstance(p["detector"]["labels"], list) or not p["detector"]["labels"]:
        raise ValueError("detector.labels must be a non-empty list")
    if not isinstance(p["constraints"]["allowed_controls"], list):
        raise ValueError("constraints.allowed_controls must be a list")
    return p


class ProfileManager:
    def __init__(self, db, log):
        self.db = db
        self.log = log

    def list(self) -> list[dict]:
        return self.db.find("profiles", sort=("name", 1))

    def get(self, pid: str) -> dict | None:
        return self.db.get("profiles", pid)

    def ensure_default(self) -> dict:
        profiles = self.list()
        if profiles:
            return profiles[0]
        return self.create(PROFILE_TEMPLATE["name"])

    def create(self, name: str) -> dict:
        p = validate({**copy.deepcopy(PROFILE_TEMPLATE), "name": name})
        p["_id"] = new_id()
        p["created"] = p["updated"] = now_iso()
        self.db.insert("profiles", p)
        self.log.info("profiles", f"Created profile '{name}'", event_id="PROFILE_CREATE")
        return p

    def save(self, profile: dict) -> dict:
        p = validate(profile)
        p["_id"] = profile["_id"]
        p["created"] = profile.get("created", now_iso())
        p["updated"] = now_iso()
        self.db.update("profiles", p["_id"], p)
        self.log.info("profiles", f"Saved profile '{p['name']}'", event_id="PROFILE_SAVE")
        return p

    def clone(self, pid: str, new_name: str) -> dict:
        src = self.get(pid)
        p = copy.deepcopy(src)
        p["_id"] = new_id()
        p["name"] = new_name
        p["model_version"] = 0
        p["active_model_id"] = ""
        p["created"] = p["updated"] = now_iso()
        self.db.insert("profiles", p)
        self.log.info("profiles", f"Cloned profile to '{new_name}'", event_id="PROFILE_CLONE")
        return p

    def delete(self, pid: str) -> None:
        self.db.delete("profiles", {"_id": pid})
        self.log.info("profiles", "Deleted profile", event_id="PROFILE_DELETE", profile_id=pid)

    def export_json(self, pid: str, path: Path) -> None:
        p = self.get(pid)
        p.pop("_id", None)
        Path(path).write_text(json.dumps(p, indent=2), encoding="utf-8")
        self.log.info("profiles", f"Exported profile to {path}", event_id="PROFILE_EXPORT")

    def import_json(self, path: Path) -> dict:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        p = validate(data)
        p["_id"] = new_id()
        p["name"] = f"{p['name']} (imported)"
        p["model_version"] = 0
        p["active_model_id"] = ""
        p["created"] = p["updated"] = now_iso()
        self.db.insert("profiles", p)
        self.log.info("profiles", f"Imported profile '{p['name']}'", event_id="PROFILE_IMPORT")
        return p
