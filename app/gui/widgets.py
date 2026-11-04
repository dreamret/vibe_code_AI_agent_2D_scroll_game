"""Reusable GUI widgets: live preview with overlays, log panel."""
from __future__ import annotations

import csv
import json
import zlib
from collections import deque

import cv2
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog, QHBoxLayout, QLineEdit,
                               QPlainTextEdit, QPushButton, QVBoxLayout, QWidget)


def label_color(label: str) -> QColor:
    return QColor.fromHsv(zlib.crc32(label.encode()) % 360, 200, 255)


class PreviewWidget(QWidget):
    """Live frame + detection overlays. Drag with the mouse to select a box
    (used by the correction panel to label a missed object)."""
    selection_made = Signal(tuple)          # (x, y, w, h) in image pixels

    def __init__(self):
        super().__init__()
        self.setMinimumSize(480, 300)
        self.setMouseTracking(True)
        self._qimg: QImage | None = None
        self._buf = None
        self.entities = []
        self.show_overlay = True
        self.capturing = False
        self.paused_text = ""
        self._drag_start: QPoint | None = None
        self._drag_now: QPoint | None = None
        self.selection: tuple | None = None   # image coords

    def set_frame(self, bgr, entities):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._buf = rgb                       # keep alive
        h, w = rgb.shape[:2]
        self._qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.entities = entities
        self.update()

    def _target_rect(self) -> QRect:
        if self._qimg is None:
            return QRect()
        iw, ih = self._qimg.width(), self._qimg.height()
        s = min(self.width() / iw, self.height() / ih)
        w, h = int(iw * s), int(ih * s)
        return QRect((self.width() - w) // 2, (self.height() - h) // 2, w, h)

    def _to_image(self, p: QPoint):
        r = self._target_rect()
        if r.width() == 0:
            return 0, 0
        s = self._qimg.width() / r.width()
        return int((p.x() - r.x()) * s), int((p.y() - r.y()) * s)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(18, 18, 22))
        if self._qimg is None:
            p.setPen(QColor(150, 150, 160))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "No capture.\nSelect a window and press 'Capture'.")
            return
        r = self._target_rect()
        p.drawImage(r, self._qimg)
        s = r.width() / self._qimg.width()
        if self.show_overlay:
            p.setFont(QFont("Segoe UI", 8))
            for e in self.entities:
                x, y, w, h = e.bbox
                col = label_color(e.label)
                p.setPen(QPen(col, 2))
                p.drawRect(QRect(r.x() + int(x * s), r.y() + int(y * s), int(w * s), int(h * s)))
                tag = f"{e.label} {e.confidence:.2f}" + (f" #{e.track_id}" if e.track_id >= 0 else "")
                p.fillRect(QRect(r.x() + int(x * s), r.y() + int(y * s) - 14, 8 * len(tag) // 1 + 6, 14),
                           QColor(0, 0, 0, 170))
                p.setPen(col)
                p.drawText(r.x() + int(x * s) + 3, r.y() + int(y * s) - 3, tag)
        if self._drag_start and self._drag_now:
            p.setPen(QPen(QColor(255, 255, 0), 2, Qt.PenStyle.DashLine))
            p.drawRect(QRect(self._drag_start, self._drag_now).normalized())
        elif self.selection:
            x, y, w, h = self.selection
            p.setPen(QPen(QColor(255, 255, 0), 2))
            p.drawRect(QRect(r.x() + int(x * s), r.y() + int(y * s), int(w * s), int(h * s)))
        if self.capturing:                    # visible capture indicator
            p.setBrush(QColor(220, 30, 30))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(10, 10, 14, 14)
            p.setPen(QColor(255, 255, 255))
            p.drawText(30, 22, "CAPTURING")
        if self.paused_text:
            p.fillRect(0, self.height() - 28, self.width(), 28, QColor(200, 120, 0, 220))
            p.setPen(QColor(255, 255, 255))
            p.drawText(10, self.height() - 9, self.paused_text)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self._qimg is not None:
            self._drag_start = self._drag_now = e.position().toPoint()

    def mouseMoveEvent(self, e):
        if self._drag_start:
            self._drag_now = e.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, e):
        if self._drag_start:
            a = self._to_image(self._drag_start)
            b = self._to_image(e.position().toPoint())
            self._drag_start = self._drag_now = None
            x, y = min(a[0], b[0]), min(a[1], b[1])
            w, h = abs(a[0] - b[0]), abs(a[1] - b[1])
            if w > 4 and h > 4:
                self.selection = (x, y, w, h)
                self.selection_made.emit(self.selection)
            self.update()


class LogPanel(QWidget):
    LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    def __init__(self):
        super().__init__()
        self.records: deque = deque(maxlen=20000)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        bar = QHBoxLayout()
        self.level = QComboBox()
        self.level.addItems(["ALL"] + self.LEVELS)
        self.level.setCurrentText("INFO")
        self.component = QComboBox()
        self.component.addItem("ALL components")
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search logs…")
        self.autoscroll_btn = QPushButton("Auto-scroll")
        self.autoscroll_btn.setCheckable(True)
        self.autoscroll_btn.setChecked(True)
        copy_btn, export_btn, clear_btn = QPushButton("Copy"), QPushButton("Export…"), QPushButton("Clear view")
        for w in (self.level, self.component, self.search, self.autoscroll_btn,
                  copy_btn, export_btn, clear_btn):
            bar.addWidget(w)
        bar.setStretch(2, 1)
        lay.addLayout(bar)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setMaximumBlockCount(5000)
        self.text.setFont(QFont("Consolas", 9))
        lay.addWidget(self.text)
        self.level.currentTextChanged.connect(self.refilter)
        self.component.currentTextChanged.connect(self.refilter)
        self.search.textChanged.connect(self.refilter)
        copy_btn.clicked.connect(self.copy)
        export_btn.clicked.connect(self.export)
        clear_btn.clicked.connect(self.clear)

    def _passes(self, r) -> bool:
        lv = self.level.currentText()
        if lv != "ALL" and self.LEVELS.index(r["level"]) < self.LEVELS.index(lv):
            return False
        comp = self.component.currentText()
        if not comp.startswith("ALL") and r["component"] != comp:
            return False
        q = self.search.text().strip().lower()
        return not q or q in self._fmt(r).lower()

    @staticmethod
    def _fmt(r) -> str:
        ctx = f"  {json.dumps(r['context'], default=str)}" if r.get("context") else ""
        return f"{r['timestamp']} [{r['level']:<8}] {r['component']}: {r['message']}{ctx}"

    def add_many(self, recs: list[dict]):
        lines = []
        for r in recs:
            self.records.append(r)
            if self.component.findText(r["component"]) < 0:
                self.component.addItem(r["component"])
            if self._passes(r):
                lines.append(self._fmt(r))
        if lines:
            self.text.appendPlainText("\n".join(lines))
            if self.autoscroll_btn.isChecked():
                sb = self.text.verticalScrollBar()
                sb.setValue(sb.maximum())

    def refilter(self, *_):
        self.text.clear()
        self.text.setPlainText("\n".join(self._fmt(r) for r in self.records if self._passes(r))[-400000:])

    def visible_text(self) -> str:
        return self.text.toPlainText()

    def copy(self):
        QApplication.clipboard().setText(self.visible_text())

    def clear(self):
        self.text.clear()

    def export(self):
        path, flt = QFileDialog.getSaveFileName(self, "Export logs", "logs.jsonl",
                                                "JSON lines (*.jsonl);;CSV (*.csv);;Text (*.txt)")
        if not path:
            return
        rows = [r for r in self.records if self._passes(r)]
        with open(path, "w", encoding="utf-8", newline="") as f:
            if path.endswith(".csv"):
                w = csv.writer(f)
                w.writerow(["timestamp", "level", "component", "event_id", "message", "context"])
                for r in rows:
                    w.writerow([r["timestamp"], r["level"], r["component"], r["event_id"],
                                r["message"], json.dumps(r["context"], default=str)])
            elif path.endswith(".txt"):
                f.write("\n".join(self._fmt(r) for r in rows))
            else:
                for r in rows:
                    f.write(json.dumps(r, default=str) + "\n")
