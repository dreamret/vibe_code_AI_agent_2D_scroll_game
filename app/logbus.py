"""Event & logging bus.

Python logging + QueueHandler/QueueListener -> (1) JSON-lines file, (2) live
subscribers (the GUI), (3) persistent `logs` collection in the database.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
from datetime import datetime
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler

_LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}


def record_to_dict(record: logging.LogRecord) -> dict:
    return {
        "timestamp": datetime.fromtimestamp(record.created).isoformat(timespec="milliseconds"),
        "level": record.levelname,
        "component": getattr(record, "component", record.name),
        "event_id": getattr(record, "event_id", ""),
        "message": record.getMessage(),
        "context": getattr(record, "context", {}) or {},
    }


class _JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps(record_to_dict(record), default=str, ensure_ascii=False)


class _CallbackHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.callbacks = []

    def emit(self, record):
        d = record_to_dict(record)
        for cb in list(self.callbacks):
            try:
                cb(d)
            except Exception:
                pass


class LogSystem:
    def __init__(self, settings):
        log_dir = settings.path("logs_dir")
        log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("aiplayersim")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        for h in list(self.logger.handlers):
            self.logger.removeHandler(h)
        self._q: queue.Queue = queue.Queue(-1)
        self.logger.addHandler(QueueHandler(self._q))
        self._cb = _CallbackHandler()
        fh = RotatingFileHandler(log_dir / "app.jsonl", maxBytes=5_000_000,
                                 backupCount=3, encoding="utf-8")
        fh.setFormatter(_JsonFormatter())
        self._listener = QueueListener(self._q, self._cb, fh, respect_handler_level=False)
        self._listener.start()
        self._store_q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._store_thread = None
        self._db = None

    # ---- subscribers ---------------------------------------------------
    def subscribe(self, callback) -> None:
        self._cb.callbacks.append(callback)

    def attach_store(self, db) -> None:
        self._db = db
        self.subscribe(lambda d: self._store_q.put(dict(d)))
        self._store_thread = threading.Thread(target=self._store_loop, daemon=True,
                                              name="log-store")
        self._store_thread.start()

    def _store_loop(self):
        while not self._stop.is_set():
            batch = []
            try:
                batch.append(self._store_q.get(timeout=1.0))
            except queue.Empty:
                continue
            while len(batch) < 300:
                try:
                    batch.append(self._store_q.get_nowait())
                except queue.Empty:
                    break
            try:
                self._db.insert_many("logs", batch)
            except Exception:
                pass

    def shutdown(self):
        self._stop.set()
        try:
            self._listener.stop()
        except Exception:
            pass

    # ---- API -----------------------------------------------------------
    def log(self, component: str, level: str, message: str, event_id: str = "", **context):
        self.logger.log(_LEVELS.get(level.upper(), 20), message,
                        extra={"component": component, "event_id": event_id,
                               "context": context})

    def debug(self, component, message, event_id="", **ctx):
        self.log(component, "DEBUG", message, event_id, **ctx)

    def info(self, component, message, event_id="", **ctx):
        self.log(component, "INFO", message, event_id, **ctx)

    def warning(self, component, message, event_id="", **ctx):
        self.log(component, "WARNING", message, event_id, **ctx)

    def error(self, component, message, event_id="", **ctx):
        self.log(component, "ERROR", message, event_id, **ctx)

    def critical(self, component, message, event_id="", **ctx):
        self.log(component, "CRITICAL", message, event_id, **ctx)
