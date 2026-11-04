"""Application settings (stored in data/settings.json)."""
from __future__ import annotations

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SETTINGS_PATH = DATA_DIR / "settings.json"

DEFAULTS = {
    # --- database -------------------------------------------------------
    "db_backend": "tinydb",              # "tinydb" (zero setup) or "mongodb"
    "mongo_uri": "mongodb://localhost:27017",
    "mongo_db": "ai_player_sim",
    "tinydb_path": "data/db/sim.json",
    # --- folders --------------------------------------------------------
    "recordings_dir": "data/recordings",
    "models_dir": "data/models",
    "templates_dir": "data/templates",
    "corrections_dir": "data/corrections",
    "logs_dir": "data/logs",
    "backup_dir": "data/backups",
    # --- capture --------------------------------------------------------
    "capture_fps": 15,
    "capture_scale": 1.0,
    "record_video": True,
    "save_frame_every_n": 15,            # sampled frame + detections stored in DB
    # --- perception -----------------------------------------------------
    "ocr_enabled": False,
    "ocr_interval_sec": 1.0,
    "tesseract_cmd": "",
    # --- automation safety ---------------------------------------------
    "emergency_hotkey": "f12",
    "max_actions_per_second": 10,
    "min_confidence": 0.45,
    "unknown_state_timeout_sec": 8.0,
    "resume_delay_sec": 5,               # get-ready countdown after pressing Resume
    "outcome_window_sec": 1.0,
    # --- logs -----------------------------------------------------------
    "log_retention_days": 30,
    # --- misc -----------------------------------------------------------
    "notice_accepted": False,
}


class Settings:
    def __init__(self, values: dict | None = None):
        self._v = dict(DEFAULTS)
        if values:
            self._v.update({k: v for k, v in values.items() if k in DEFAULTS})

    def __getitem__(self, key):
        return self._v[key]

    def __setitem__(self, key, value):
        self._v[key] = value

    def get(self, key, default=None):
        return self._v.get(key, default)

    def as_dict(self) -> dict:
        return dict(self._v)

    def path(self, key: str) -> Path:
        p = Path(self._v[key])
        return p if p.is_absolute() else BASE_DIR / p

    @classmethod
    def load(cls) -> "Settings":
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if SETTINGS_PATH.exists():
            try:
                return cls(json.loads(SETTINGS_PATH.read_text(encoding="utf-8")))
            except Exception:
                pass
        s = cls()
        s.save()
        return s

    def save(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps(self._v, indent=2), encoding="utf-8")

    def ensure_dirs(self) -> None:
        for key in ("recordings_dir", "models_dir", "templates_dir",
                    "corrections_dir", "logs_dir", "backup_dir"):
            self.path(key).mkdir(parents=True, exist_ok=True)
        self.path("tinydb_path").parent.mkdir(parents=True, exist_ok=True)
