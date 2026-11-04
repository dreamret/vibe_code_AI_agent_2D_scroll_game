"""Local NoSQL data layer.

Two interchangeable document backends behind one API:
  * TinyDB  - embedded JSON document store, zero setup (default / prototype)
  * MongoDB - local Community Server, recommended for large datasets

Collections: profiles, sessions, frames, entities, actions,
             training_examples, models, logs
Large media (video, JPEG frames) and model weights stay on disk; the DB keeps
paths, hashes and metadata.
"""
from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

COLLECTIONS = ["profiles", "sessions", "frames", "entities", "actions",
               "training_examples", "models", "logs"]

# (collection, field(s)) indexes created on MongoDB
MONGO_INDEXES = {
    "sessions": ["start_time"],
    "frames": ["session_id", "timestamp"],
    "entities": ["session_id", "frame_id", "label"],
    "actions": ["session_id", "timestamp"],
    "training_examples": ["profile_id", "review_status", "source_session"],
    "models": ["profile_id", "version"],
    "logs": ["timestamp", "level", "component"],
}


def new_id() -> str:
    return uuid.uuid4().hex


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def clean(o):
    """Convert to plain JSON/BSON-safe types (numpy, Path, tuples...)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [clean(v) for v in o]
    if o is None or isinstance(o, (str, int, float, bool)):
        return o
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    return str(o)


def match(doc: dict, query: dict | None) -> bool:
    """Tiny query matcher (equality + $in/$ne/$gt/$gte/$lt/$lte)."""
    if not query:
        return True
    for key, cond in query.items():
        val = doc.get(key)
        if isinstance(cond, dict) and any(k.startswith("$") for k in cond):
            for op, ref in cond.items():
                try:
                    if op == "$in" and val not in ref: return False
                    if op == "$ne" and val == ref: return False
                    if op == "$gt" and not (val is not None and val > ref): return False
                    if op == "$gte" and not (val is not None and val >= ref): return False
                    if op == "$lt" and not (val is not None and val < ref): return False
                    if op == "$lte" and not (val is not None and val <= ref): return False
                except TypeError:
                    return False
        elif val != cond:
            return False
    return True


class _TinyBackend:
    name = "tinydb"

    def __init__(self, path: Path):
        from tinydb import TinyDB
        from tinydb.middlewares import CachingMiddleware
        from tinydb.storages import JSONStorage
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._db = TinyDB(str(path), storage=CachingMiddleware(JSONStorage))

    def insert(self, coll, doc):
        self._db.table(coll).insert(doc)

    def insert_many(self, coll, docs):
        self._db.table(coll).insert_multiple(docs)

    def find(self, coll, query, limit, sort):
        rows = [dict(d) for d in self._db.table(coll).all() if match(d, query)]
        if sort:
            field, direction = sort
            rows.sort(key=lambda d: (d.get(field) is None, d.get(field)),
                      reverse=(direction < 0))
        return rows[:limit] if limit else rows

    def update(self, coll, _id, changes):
        self._db.table(coll).update(changes, lambda d: d.get("_id") == _id)

    def delete(self, coll, query):
        return len(self._db.table(coll).remove(lambda d: match(d, query)))

    def count(self, coll, query):
        if not query:
            return len(self._db.table(coll))
        return sum(1 for d in self._db.table(coll).all() if match(d, query))

    def ensure_indexes(self):
        pass  # TinyDB has no indexes

    def flush(self):
        try:
            self._db.storage.flush()
        except Exception:
            pass

    def close(self):
        self.flush()
        self._db.close()


class _MongoBackend:
    name = "mongodb"

    def __init__(self, uri: str, dbname: str):
        from pymongo import MongoClient
        self._client = MongoClient(uri, serverSelectionTimeoutMS=2000)
        self._client.admin.command("ping")          # raises if unreachable
        self._db = self._client[dbname]

    def insert(self, coll, doc):
        self._db[coll].insert_one(doc)

    def insert_many(self, coll, docs):
        if docs:
            self._db[coll].insert_many(docs)

    def find(self, coll, query, limit, sort):
        cur = self._db[coll].find(query or {})
        if sort:
            cur = cur.sort(sort[0], sort[1])
        if limit:
            cur = cur.limit(limit)
        return list(cur)

    def update(self, coll, _id, changes):
        self._db[coll].update_one({"_id": _id}, {"$set": changes})

    def delete(self, coll, query):
        return self._db[coll].delete_many(query or {}).deleted_count

    def count(self, coll, query):
        return self._db[coll].count_documents(query or {})

    def ensure_indexes(self):
        for coll, fields in MONGO_INDEXES.items():
            for f in fields:
                self._db[coll].create_index(f)

    def flush(self):
        pass

    def close(self):
        self._client.close()


class Database:
    def __init__(self, settings, log):
        self.settings = settings
        self.log = log
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self.backend = self._open_backend()
        self.backend.ensure_indexes()
        self.log.info("database", f"Database ready ({self.backend.name})",
                      event_id="DB_READY")
        threading.Thread(target=self._flush_loop, daemon=True, name="db-flush").start()

    def _open_backend(self):
        want = self.settings["db_backend"]
        if want == "mongodb":
            try:
                return _MongoBackend(self.settings["mongo_uri"], self.settings["mongo_db"])
            except Exception as exc:
                self.log.warning("database",
                                 f"MongoDB unavailable ({exc.__class__.__name__}); "
                                 "falling back to embedded TinyDB",
                                 event_id="DB_FALLBACK")
        try:
            return _TinyBackend(self.settings.path("tinydb_path"))
        except ImportError as exc:
            raise RuntimeError("No database backend available. Run: pip install tinydb "
                               "(or pymongo + a local MongoDB server).") from exc

    @property
    def backend_name(self) -> str:
        return self.backend.name

    def _flush_loop(self):
        while not self._stop.wait(5.0):
            with self._lock:
                self.backend.flush()

    # ---- CRUD ----------------------------------------------------------
    def insert(self, coll: str, doc: dict) -> str:
        doc = clean(doc)
        doc.setdefault("_id", new_id())
        with self._lock:
            self.backend.insert(coll, doc)
        return doc["_id"]

    def insert_many(self, coll: str, docs: list[dict]) -> list[str]:
        docs = [clean(d) for d in docs]
        for d in docs:
            d.setdefault("_id", new_id())
        if docs:
            with self._lock:
                self.backend.insert_many(coll, docs)
        return [d["_id"] for d in docs]

    def find(self, coll: str, query: dict | None = None, limit: int = 0,
             sort: tuple | None = None) -> list[dict]:
        with self._lock:
            return self.backend.find(coll, clean(query or {}), limit, sort)

    def get(self, coll: str, _id: str) -> dict | None:
        rows = self.find(coll, {"_id": _id}, limit=1)
        return rows[0] if rows else None

    def update(self, coll: str, _id: str, changes: dict) -> None:
        with self._lock:
            self.backend.update(coll, _id, clean(changes))

    def delete(self, coll: str, query: dict) -> int:
        with self._lock:
            return self.backend.delete(coll, clean(query))

    def count(self, coll: str, query: dict | None = None) -> int:
        with self._lock:
            return self.backend.count(coll, clean(query or {}))

    def flush(self):
        with self._lock:
            self.backend.flush()

    # ---- retention / backup / export ----------------------------------
    def purge_logs(self, days: int) -> int:
        cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="milliseconds")
        n = self.delete("logs", {"timestamp": {"$lt": cutoff}})
        self.log.info("database", f"Purged {n} log records older than {days} days",
                      event_id="LOG_PURGE")
        return n

    def delete_session(self, session_id: str, remove_files: bool = True) -> None:
        for coll in ("frames", "entities", "actions", "training_examples"):
            key = "source_session" if coll == "training_examples" else "session_id"
            self.delete(coll, {key: session_id})
        self.delete("sessions", {"_id": session_id})
        if remove_files:
            shutil.rmtree(self.settings.path("recordings_dir") / session_id,
                          ignore_errors=True)
        self.log.info("database", f"Deleted session {session_id}", event_id="SESSION_DELETE")

    def export_json(self, dest: Path, include_logs: bool = True) -> Path:
        out = {}
        for coll in COLLECTIONS:
            if coll == "logs" and not include_logs:
                continue
            out[coll] = self.find(coll)
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
        self.log.info("database", f"Exported database to {dest}", event_id="DB_EXPORT")
        return dest

    def backup(self) -> Path:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        dest_dir = self.settings.path("backup_dir")
        dest_dir.mkdir(parents=True, exist_ok=True)
        if self.backend.name == "tinydb":
            self.flush()
            dest = dest_dir / f"sim_{stamp}.json"
            shutil.copy2(self.backend.path, dest)
        else:
            dest = self.export_json(dest_dir / f"mongo_{stamp}.json")
        self.log.info("database", f"Backup written to {dest}", event_id="DB_BACKUP")
        return dest

    def close(self):
        self._stop.set()
        with self._lock:
            self.backend.close()
