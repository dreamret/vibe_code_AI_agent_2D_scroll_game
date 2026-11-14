"""Extra GUI panels: auto-label review, elements, media, database browser, keybind dialog."""
from __future__ import annotations

import json
import os
import threading

import cv2
from PySide6.QtCore import QObject, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QFont, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSlider,
                               QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .. import media as media_mod
from ..db import COLLECTIONS
from .widgets import PreviewWidget


def bgr_to_icon(img, size=48) -> QIcon:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qi = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    return QIcon(QPixmap.fromImage(qi).scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio,
                                              Qt.TransformationMode.SmoothTransformation))


def _btn(text, fn=None):
    b = QPushButton(text)
    if fn:
        b.clicked.connect(fn)
    return b


class JsonEditDialog(QDialog):
    def __init__(self, parent, title, text):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(640, 640)
        lay = QVBoxLayout(self)
        self.edit = QPlainTextEdit(text)
        self.edit.setFont(QFont("Consolas", 10))
        lay.addWidget(self.edit)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)


# ------------------------------------------------------------------ auto-label review
class ClusterReview(QWidget):
    """Groups of look-alike unlabelled elements. Nothing is saved until you tick + Apply."""
    applied = Signal()

    def __init__(self, ctrl):
        super().__init__()
        self.ctrl = ctrl
        self.clusters: list[dict] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.lbl = QLabel("No suggestions yet.")
        self.lbl.setWordWrap(True)
        lay.addWidget(self.lbl)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Apply", "Sample", "Seen", "Looks like", "Label"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(3, 190)
        self.table.setIconSize(QSize(48, 48))
        self.table.verticalHeader().setDefaultSectionSize(54)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        lay.addWidget(self.table, 1)
        row = QHBoxLayout()
        row.addWidget(_btn("Tick all", lambda: self._tick(True)))
        row.addWidget(_btn("Untick all", lambda: self._tick(False)))
        row.addStretch(1)
        row.addWidget(_btn("Apply ticked labels", self.apply))
        lay.addLayout(row)

    def label_choices(self) -> list[str]:
        names = list(self.ctrl.profile["detector"]["labels"])
        names += [e["label"] for e in self.ctrl.list_elements() if e["label"] not in names]
        return names

    def set_clusters(self, clusters: list[dict], source: str = ""):
        self.clusters = clusters
        self.table.setRowCount(len(clusters))
        choices = self.label_choices()
        for r, c in enumerate(clusters):
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            chk.setCheckState(Qt.CheckState.Checked if c["known"] else Qt.CheckState.Unchecked)
            self.table.setItem(r, 0, chk)
            icon_item = QTableWidgetItem()
            icon_item.setIcon(bgr_to_icon(c["samples"][0]["crop"]))
            self.table.setItem(r, 1, icon_item)
            self.table.setItem(r, 2, QTableWidgetItem(str(c["count"])))
            known = f"matches '{c['suggested_label']}' ({c['match_confidence']:.2f})" if c["known"] \
                else f"new: {c['hint']}"
            self.table.setItem(r, 3, QTableWidgetItem(
                f"{known}\nmoving {c['moving_ratio']:.0%}, {c['size'][0]}x{c['size'][1]}px"))
            cb = QComboBox()
            cb.setEditable(True)
            cb.addItems(choices)
            cb.setCurrentText(c["suggested_label"])
            self.table.setCellWidget(r, 4, cb)
        self.lbl.setText(f"{len(clusters)} group(s) from {source}. Tick the ones you want, correct the "
                         "label if needed (type a new name to create a new element), then Apply."
                         if clusters else f"Nothing recurring found in {source}. Try a longer scan "
                         "with more movement on screen.")

    def _tick(self, on):
        for r in range(self.table.rowCount()):
            self.table.item(r, 0).setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)

    def apply(self):
        sel, rows = [], []
        for r, c in enumerate(self.clusters):
            if self.table.item(r, 0).checkState() == Qt.CheckState.Checked:
                label = self.table.cellWidget(r, 4).currentText().strip()
                if label:
                    sel.append((c, label))
                    rows.append(r)
        if not sel:
            QMessageBox.information(self, "Auto-label", "Tick at least one group.")
            return
        n = self.ctrl.apply_clusters(sel)
        for r in reversed(rows):
            self.table.removeRow(r)
            del self.clusters[r]
        self.lbl.setText(f"Applied {n} group(s): templates + element entries saved.")
        self.applied.emit()


# ------------------------------------------------------------------ elements
class ElementDialog(QDialog):
    def __init__(self, parent, categories, data=None):
        super().__init__(parent)
        self.setWindowTitle("Element")
        data = data or {}
        f = QFormLayout(self)
        self.label = QLineEdit(data.get("label", ""))
        self.label.setPlaceholderText("unique id, e.g. slime / 史萊姆 / ポーション")
        self.name = QLineEdit(data.get("display_name", ""))
        self.cat = QComboBox()
        self.cat.addItems(categories)
        self.cat.setCurrentText(data.get("category", "other"))
        self.enabled = QCheckBox("Detect this element")
        self.enabled.setChecked(data.get("enabled", True))
        self.notes = QPlainTextEdit(data.get("description", ""))
        self.notes.setMaximumHeight(90)
        f.addRow("Label:", self.label)
        f.addRow("Display name:", self.name)
        f.addRow("Category:", self.cat)
        f.addRow(self.enabled)
        f.addRow("Notes:", self.notes)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)

    def data(self) -> dict:
        return {"label": self.label.text(), "display_name": self.name.text() or self.label.text(),
                "category": self.cat.currentText(), "enabled": self.enabled.isChecked(),
                "description": self.notes.toPlainText()}


class ElementsPanel(QWidget):
    changed = Signal()

    def __init__(self, ctrl):
        super().__init__()
        self.ctrl = ctrl
        self._rows: list[dict] = []
        lay = QVBoxLayout(self)
        info = QLabel("Elements are the things the agent looks for (characters, enemies, items, UI, "
                      "markers…). Each element is a label with example images (templates).")
        info.setWordWrap(True)
        lay.addWidget(info)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Label", "Name", "Category", "On", "Templates", "Notes"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.doubleClicked.connect(self.edit)
        lay.addWidget(self.table, 1)
        row = QHBoxLayout()
        for text, fn in (("Add…", self.add), ("Edit…", self.edit), ("Delete", self.delete),
                         ("Add template images…", self.add_images), ("Open folder", self.open_folder),
                         ("Refresh", self.refresh)):
            row.addWidget(_btn(text, fn))
        lay.addLayout(row)

    def refresh(self):
        self._rows = self.ctrl.list_elements()
        self.table.setRowCount(len(self._rows))
        for r, e in enumerate(self._rows):
            vals = [e["label"], e.get("display_name", ""), e.get("category", ""),
                    "✔" if e.get("enabled", True) else "—", self.ctrl.element_template_count(e["label"]),
                    e.get("description", "")]
            for c, v in enumerate(vals):
                self.table.setItem(r, c, QTableWidgetItem(str(v)))

    def _current(self):
        rows = self.table.selectionModel().selectedRows()
        return self._rows[rows[0].row()] if rows else None

    def add(self):
        d = ElementDialog(self, self.ctrl.ELEMENT_CATEGORIES)
        if d.exec() == QDialog.DialogCode.Accepted and d.data()["label"].strip():
            self.ctrl.save_element(d.data())
            self.refresh()
            self.changed.emit()

    def edit(self, *_):
        cur = self._current()
        if not cur:
            return
        d = ElementDialog(self, self.ctrl.ELEMENT_CATEGORIES, cur)
        if d.exec() == QDialog.DialogCode.Accepted:
            self.ctrl.save_element({**cur, **d.data(), "label": cur["label"]})   # label is the key
            self.refresh()
            self.changed.emit()

    def delete(self):
        cur = self._current()
        if not cur:
            return
        r = QMessageBox.question(self, "Delete element", f"Delete '{cur['label']}'?\n\nYes = also delete "
                                 "its template images. No = keep the images.",
                                 QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No |
                                 QMessageBox.StandardButton.Cancel)
        if r == QMessageBox.StandardButton.Cancel:
            return
        self.ctrl.delete_element(cur["_id"], remove_templates=(r == QMessageBox.StandardButton.Yes))
        self.refresh()

    def add_images(self):
        cur = self._current()
        if not cur:
            QMessageBox.information(self, "Templates", "Select an element first.")
            return
        files, _ = QFileDialog.getOpenFileNames(self, "Template images (crops of this element)", "",
                                                "Images (*.png *.jpg *.jpeg *.bmp)")
        if files:
            n = self.ctrl.add_element_templates(cur["label"], files)
            QMessageBox.information(self, "Templates", f"Imported {n} image(s).")
            self.refresh()

    def open_folder(self):
        cur = self._current()
        if not cur:
            return
        d = self.ctrl.settings.path("templates_dir") / self.ctrl.profile["_id"] / cur["label"]
        d.mkdir(parents=True, exist_ok=True)
        if hasattr(os, "startfile"):
            os.startfile(str(d))


# ------------------------------------------------------------------ media
class _MediaSignals(QObject):
    progress = Signal(int, int)
    done = Signal(object)
    failed = Signal(str)


class MediaPanel(QWidget):
    def __init__(self, ctrl):
        super().__init__()
        self.ctrl = ctrl
        self.result = None
        self._stop = threading.Event()
        self.sig = _MediaSignals()
        self.sig.progress.connect(lambda i, n: (self.bar.setMaximum(n), self.bar.setValue(i)))
        self.sig.done.connect(self._on_done)
        self.sig.failed.connect(self._on_failed)
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.btn_open = _btn("Open GIF / video / image…", self.open_media)
        self.btn_stop = _btn("Cancel", self._stop.set)
        self.bar = QProgressBar()
        top.addWidget(self.btn_open)
        top.addWidget(self.bar, 1)
        top.addWidget(self.btn_stop)
        lay.addLayout(top)
        self.preview = PreviewWidget()
        self.preview.setMinimumSize(320, 200)
        lay.addWidget(self.preview, 3)
        row = QHBoxLayout()
        self.btn_play = _btn("▶", self.toggle_play)
        self.btn_play.setMaximumWidth(36)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.show_frame)
        self.lbl_t = QLabel("")
        row.addWidget(self.btn_play)
        row.addWidget(self.slider, 1)
        row.addWidget(self.lbl_t)
        lay.addLayout(row)
        self.txt = QPlainTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setMaximumHeight(150)
        lay.addWidget(self.txt)
        self.review = ClusterReview(ctrl)
        lay.addWidget(QLabel("Suggested labels for unknown recurring elements:"))
        lay.addWidget(self.review, 2)
        ex = QHBoxLayout()
        ex.addWidget(_btn("Export annotated GIF/MP4…", self.export_annotated))
        ex.addWidget(_btn("Export report (JSON)…", self.export_report))
        lay.addLayout(ex)
        self.timer = QTimer(self, interval=150, timeout=self._advance)

    def open_media(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open media", "", media_mod.MEDIA_FILTER)
        if not path:
            return
        self._stop.clear()
        self.btn_open.setEnabled(False)
        self.txt.setPlainText("Analysing… (the first run loads the detectors)")

        def run():
            try:
                res = self.ctrl.analyze_media(path, progress=self.sig.progress.emit, stop=self._stop)
                self.sig.done.emit(res)
            except Exception as exc:
                self.sig.failed.emit(str(exc))
        threading.Thread(target=run, daemon=True, name="media").start()

    def _on_failed(self, msg):
        self.btn_open.setEnabled(True)
        QMessageBox.warning(self, "Media analysis failed", msg)

    def _on_done(self, res):
        self.btn_open.setEnabled(True)
        self.timer.stop()
        self.btn_play.setText("▶")
        self.result = res
        self.txt.setPlainText(media_mod.summary_text(res))
        self.slider.blockSignals(True)
        self.slider.setRange(0, len(res.frames) - 1)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.show_frame(0)
        self.review.set_clusters(res.clusters, res.info["name"])

    def show_frame(self, i):
        r = self.result
        if not r or i >= len(r.frames):
            return
        self.preview.set_frame(r.frames[i], r.entities[i])
        self.lbl_t.setText(f"{r.times[i]:.2f}s  ({i + 1}/{len(r.frames)})")

    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_play.setText("▶")
        elif self.result and len(self.result.frames) > 1:
            self.timer.start()
            self.btn_play.setText("❚❚")

    def _advance(self):
        n = len(self.result.frames)
        self.slider.setValue((self.slider.value() + 1) % n)

    def export_annotated(self):
        if not self.result:
            return
        default = "annotated.gif" if self.result.info["type"] == "gif" else "annotated.mp4"
        path, _ = QFileDialog.getSaveFileName(self, "Export annotated", default,
                                              "GIF (*.gif);;MP4 (*.mp4);;PNG (*.png)")
        if path:
            try:
                media_mod.export_annotated(self.result, path)
            except Exception as exc:
                QMessageBox.warning(self, "Export failed", str(exc))

    def export_report(self):
        if not self.result:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export report", "media_report.json", "JSON (*.json)")
        if path:
            media_mod.export_report(self.result, self.ctrl.profile["_id"], path)


# ------------------------------------------------------------------ database browser
SORTS = {"logs": ("timestamp", -1), "frames": ("timestamp", -1), "sessions": ("start_time", -1),
         "actions": ("timestamp", -1), "training_examples": ("created", -1), "models": ("version", -1),
         "profiles": ("name", 1), "entities": ("timestamp", -1), "elements": ("label", 1),
         "keybinds": ("created", -1), "media": ("created", -1)}
PROFILE_SCOPED = {"elements", "keybinds", "training_examples", "models", "media", "sessions"}
ROW_LIMIT = 500


class DbBrowser(QWidget):
    changed = Signal(str)

    def __init__(self, ctrl):
        super().__init__()
        self.ctrl = ctrl
        self._rows: list[dict] = []
        lay = QVBoxLayout(self)
        warn = QLabel("Advanced: direct edits bypass validation. Use Data → Backup first if unsure.")
        warn.setStyleSheet("color:#b36b00")
        lay.addWidget(warn)
        top = QHBoxLayout()
        self.cmb = QComboBox()
        self.cmb.addItems(COLLECTIONS)
        self.cmb.setCurrentText("elements")
        self.cmb.currentTextChanged.connect(self.refresh)
        self.chk_profile = QCheckBox("current profile only")
        self.chk_profile.setChecked(True)
        self.chk_profile.toggled.connect(self.refresh)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search rows…")
        self.search.textChanged.connect(self._fill)
        self.lbl = QLabel("")
        for w in (QLabel("Collection:"), self.cmb, self.chk_profile, self.search, self.lbl):
            top.addWidget(w)
        top.setStretch(3, 1)
        lay.addLayout(top)
        self.table = QTableWidget(0, 0)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.doubleClicked.connect(self.edit)
        lay.addWidget(self.table, 1)
        row = QHBoxLayout()
        for text, fn in (("Refresh", self.refresh), ("New…", self.new), ("Edit…", self.edit),
                         ("Delete selected", self.delete), ("Export view…", self.export_view)):
            row.addWidget(_btn(text, fn))
        lay.addLayout(row)

    def refresh(self, *_):
        coll = self.cmb.currentText()
        q = {}
        scoped = coll in PROFILE_SCOPED
        self.chk_profile.setEnabled(scoped)
        if scoped and self.chk_profile.isChecked():
            q["profile_id"] = self.ctrl.profile["_id"]
        self._rows = self.ctrl.db.find(coll, q, limit=ROW_LIMIT, sort=SORTS.get(coll))
        self._fill()

    @staticmethod
    def _fmt(v) -> str:
        if isinstance(v, dict):
            return f"{{{len(v)} keys}}"
        if isinstance(v, list):
            return f"[{len(v)}]"
        s = str(v)
        return s if len(s) <= 60 else s[:57] + "…"

    def _fill(self, *_):
        q = self.search.text().strip().lower()
        rows = [r for r in self._rows if not q or q in json.dumps(r, default=str, ensure_ascii=False).lower()]
        self._shown = rows
        keys: list[str] = []
        for r in rows[:40]:
            for k in r:
                if k != "_id" and k not in keys:
                    keys.append(k)
        keys = keys[:9]
        self.table.setColumnCount(len(keys) + 1)
        self.table.setHorizontalHeaderLabels(["_id"] + keys)
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            it = QTableWidgetItem(str(r.get("_id", ""))[:10])
            it.setData(Qt.ItemDataRole.UserRole, r.get("_id"))
            self.table.setItem(i, 0, it)
            for c, k in enumerate(keys, 1):
                self.table.setItem(i, c, QTableWidgetItem(self._fmt(r.get(k, ""))))
        self.lbl.setText(f"{len(rows)} shown (max {ROW_LIMIT}; total {self.ctrl.db.count(self.cmb.currentText())})")

    def _selected(self) -> list[dict]:
        return [self._shown[i.row()] for i in self.table.selectionModel().selectedRows()]

    def new(self):
        coll = self.cmb.currentText()
        tpl = {"profile_id": self.ctrl.profile["_id"]} if coll in PROFILE_SCOPED else {}
        if coll == "elements":
            tpl.update(label="", display_name="", category="other", enabled=True, description="")
        d = JsonEditDialog(self, f"New document in {coll}", json.dumps(tpl, indent=2, ensure_ascii=False))
        if d.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            doc = json.loads(d.edit.toPlainText())
            if not isinstance(doc, dict):
                raise ValueError("A document must be a JSON object")
            if coll == "elements":
                self.ctrl.save_element(doc)
            else:
                self.ctrl.db.insert(coll, doc)
                self.changed.emit(coll)
            self.refresh()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid document", str(exc))

    def edit(self, *_):
        sel = self._selected()
        if not sel:
            return
        coll, doc = self.cmb.currentText(), sel[0]
        d = JsonEditDialog(self, f"Edit {coll} / {doc['_id'][:8]}",
                           json.dumps(doc, indent=2, ensure_ascii=False, default=str))
        if d.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            new = json.loads(d.edit.toPlainText())
            if not isinstance(new, dict):
                raise ValueError("A document must be a JSON object")
            if new.get("_id") != doc["_id"]:
                raise ValueError("Do not change _id")
            self.ctrl.db.replace(coll, doc["_id"], new)
            self.ctrl.log.info("database", f"Edited {coll} document via GUI", event_id="DB_EDIT")
            self.changed.emit(coll)
            self.refresh()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid document", str(exc))

    def delete(self):
        sel = self._selected()
        if not sel:
            return
        coll = self.cmb.currentText()
        if QMessageBox.question(self, "Delete", f"Delete {len(sel)} document(s) from '{coll}'?") != \
                QMessageBox.StandardButton.Yes:
            return
        for d in sel:
            self.ctrl.db.delete(coll, {"_id": d["_id"]})
        self.ctrl.log.info("database", f"Deleted {len(sel)} {coll} document(s) via GUI", event_id="DB_DELETE")
        self.changed.emit(coll)
        self.refresh()

    def export_view(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export", f"{self.cmb.currentText()}.json", "JSON (*.json)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._shown, f, indent=1, ensure_ascii=False, default=str)


# ------------------------------------------------------------------ keybind dialog
class KeybindDialog(QDialog):
    """The agent compares the demo with the profile's keybinds and asks about every
    key that was used in the demo but has no keybind."""

    def __init__(self, parent, analysis: dict, title="Keybinds found in the demo"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(940, 460)
        self.a = analysis
        lay = QVBoxLayout(self)
        un = analysis["unmapped"]
        head = (f"The demo used {len(un)} key(s)/button(s) that have no keybind in this profile. "
                "What is the purpose of each action?" if un else
                "Every key used in the demo already has a keybind in this profile.")
        lbl = QLabel(head)
        lbl.setWordWrap(True)
        lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        lay.addWidget(lbl)
        if analysis["used_mapped"] or analysis["unused_mapped"]:
            lay.addWidget(QLabel("Mapped & used: " + (", ".join(analysis["used_mapped"]) or "—")
                                 + "\nMapped but NOT used in this demo: "
                                 + (", ".join(analysis["unused_mapped"]) or "—")))
        self.table = QTableWidget(len(un), 6)
        self.table.setHorizontalHeaderLabels(["Key / button", "Used", "What happened after it",
                                              "Action name", "Purpose (your words)", "Decision"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(2, 260)
        self.table.setColumnWidth(3, 130)
        self.table.setColumnWidth(4, 170)
        self.table.verticalHeader().setDefaultSectionSize(34)
        self._widgets = []
        for r, u in enumerate(un):
            self.table.setItem(r, 0, QTableWidgetItem(f"{u['control']}  ({u['device']})"))
            self.table.setItem(r, 1, QTableWidgetItem(f"{u['count']}×  avg {u['avg_duration']}s"))
            self.table.setItem(r, 2, QTableWidgetItem(u["hint"]))
            name = QComboBox()
            name.setEditable(True)
            name.addItems([""] + analysis["action_names"])
            purpose = QLineEdit()
            purpose.setPlaceholderText("e.g. drink HP potion")
            dec = QComboBox()
            dec.addItems(["Add to profile", "Ignore (never replay)", "Ask me later"])
            for c, w in ((3, name), (4, purpose), (5, dec)):
                self.table.setCellWidget(r, c, w)
            self._widgets.append((u, name, purpose, dec))
        lay.addWidget(self.table, 1)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Close)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def answers(self) -> list[dict]:
        out = []
        for u, name, purpose, dec in self._widgets:
            out.append({"control": u["control"], "device": u["device"],
                        "decision": ["add", "ignore", "later"][dec.currentIndex()],
                        "action_name": name.currentText().strip(), "purpose": purpose.text().strip()})
        return out
