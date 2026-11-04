"""Control Center main window (PySide6)."""
from __future__ import annotations

import json
import os
import time
from collections import deque
from datetime import datetime

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QFont
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog,
                               QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
                               QListWidget, QMainWindow, QMenu, QMessageBox, QPlainTextEdit,
                               QPushButton, QSpinBox, QSplitter, QTableWidget,
                               QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from .. import APP_NAME, winutil
from ..controller import AppController
from .widgets import LogPanel, PreviewWidget

STATE_STYLE = {
    "OFF": "background:#555;color:white",
    "ARMED": "background:#1b8a3a;color:white",
    "PAUSED": "background:#d98200;color:white",
    "EMERGENCY": "background:#c0141c;color:white",
}


class Bridge(QObject):
    state = Signal()
    proposal = Signal(object)
    action = Signal(object)
    notice = Signal(str, str)


class JsonDialog(QDialog):
    def __init__(self, parent, title, text):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(640, 640)
        lay = QVBoxLayout(self)
        self.edit = QPlainTextEdit(text)
        self.edit.setFont(QFont("Consolas", 10))
        lay.addWidget(self.edit)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Save |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)


class MainWindow(QMainWindow):
    def __init__(self, ctrl: AppController):
        super().__init__()
        self.ctrl = ctrl
        self.setWindowTitle(APP_NAME)
        self.resize(1500, 920)
        self._log_queue: deque = deque()
        self._last_frame_idx = -1
        self._frozen = None
        self._warned_execution = False

        self.bridge = Bridge()
        ctrl.on_state = self.bridge.state.emit
        ctrl.on_proposal = self.bridge.proposal.emit
        ctrl.on_action = self.bridge.action.emit
        ctrl.on_notice = self.bridge.notice.emit
        ctrl.log.subscribe(self._log_queue.append)
        self.bridge.state.connect(self.refresh_state)
        self.bridge.proposal.connect(self.show_proposal)
        self.bridge.action.connect(self.add_action_row)
        self.bridge.notice.connect(lambda t, m: QMessageBox.warning(self, t, m))

        self._build_menu()
        self._build_ui()
        self.refresh_windows()
        self.refresh_profiles()
        self.refresh_state()
        self.refresh_learning()

        self.t_fast = QTimer(self, interval=33, timeout=self.tick_preview)
        self.t_logs = QTimer(self, interval=150, timeout=self.drain_logs)
        self.t_info = QTimer(self, interval=400, timeout=self.tick_info)
        for t in (self.t_fast, self.t_logs, self.t_info):
            t.start()

    # ------------------------------------------------------------------ UI
    def _btn(self, text, checkable=False, tip=""):
        b = QPushButton(text)
        b.setCheckable(checkable)
        b.setToolTip(tip)
        return b

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        v = QVBoxLayout(root)

        # ---- top bar ----
        top = QHBoxLayout()
        self.cmb_window = QComboBox()
        self.cmb_window.setMinimumWidth(280)
        self.cmb_window.currentIndexChanged.connect(self.on_window_changed)
        b_ref = self._btn("⟳", tip="Refresh window list")
        b_ref.setMaximumWidth(34)
        b_ref.clicked.connect(self.refresh_windows)
        self.cmb_profile = QComboBox()
        self.cmb_profile.setMinimumWidth(200)
        self.cmb_profile.currentIndexChanged.connect(self.on_profile_changed)
        self.btn_capture = self._btn("Capture", True, "Start / pause screen capture")
        self.btn_record = self._btn("Record", True, "Record video, frames and your inputs")
        self.btn_learn = self._btn("Learn", True, "Create reviewable training examples while recording")
        self.btn_assist = self._btn("Assist / Replay ▾")
        self.btn_assist.setMenu(self._build_assist_menu())
        self.btn_resume = self._btn("Resume", tip="Resume after you handled a pause (e.g. CAPTCHA)")
        self.btn_stop = self._btn("■ STOP", tip="Stop everything and disable input execution")
        self.btn_stop.setStyleSheet("background:#c0141c;color:white;font-weight:bold")
        self.btn_capture.toggled.connect(self.on_capture_toggled)
        self.btn_record.toggled.connect(self.on_record_toggled)
        self.btn_learn.toggled.connect(self.ctrl.set_learn)
        self.btn_resume.clicked.connect(self.ctrl.resume)
        self.btn_stop.clicked.connect(self.on_stop)
        top.addWidget(QLabel("Target:"))
        top.addWidget(self.cmb_window)
        top.addWidget(b_ref)
        top.addWidget(QLabel("Profile:"))
        top.addWidget(self.cmb_profile)
        for b in (self.btn_capture, self.btn_record, self.btn_learn, self.btn_assist,
                  self.btn_resume, self.btn_stop):
            top.addWidget(b)
        top.addStretch(1)
        self.lbl_auto = QLabel("AUTOMATION: OFF")
        self.lbl_auto.setMinimumWidth(240)
        self.lbl_auto.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_auto.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        top.addWidget(self.lbl_auto)
        v.addLayout(top)

        # ---- centre ----
        self.preview = PreviewWidget()
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_scene_tab(), "Scene")
        self.tabs.addTab(self._build_action_tab(), "Actions")
        self.tabs.addTab(self._build_correction_tab(), "Corrections")
        self.tabs.addTab(self._build_learning_tab(), "Learning")
        self.tabs.addTab(self._build_settings_tab(), "Settings")
        hs = QSplitter(Qt.Orientation.Horizontal)
        hs.addWidget(self.preview)
        hs.addWidget(self.tabs)
        hs.setStretchFactor(0, 3)
        hs.setStretchFactor(1, 2)
        self.logs = LogPanel()
        vs = QSplitter(Qt.Orientation.Vertical)
        vs.addWidget(hs)
        vs.addWidget(self.logs)
        vs.setStretchFactor(0, 3)
        vs.setStretchFactor(1, 1)
        v.addWidget(vs, 1)

        self.lbl_status = QLabel()
        self.statusBar().addWidget(self.lbl_status, 1)
        self.statusBar().addPermanentWidget(QLabel(f"DB: {self.ctrl.db.backend_name}   "))

    def _build_assist_menu(self) -> QMenu:
        m = QMenu(self)
        self.act_exec = QAction("Enable input execution (human switch)", self, checkable=True)
        self.act_exec.toggled.connect(self.on_exec_toggled)
        self.act_assist = QAction("Assist mode: propose actions for my approval", self,
                                  checkable=True)
        self.act_assist.toggled.connect(self.on_assist_toggled)
        act_replay = QAction("Replay a recorded session…", self)
        act_replay.triggered.connect(self.on_replay)
        act_rearm = QAction("Clear emergency stop", self)
        act_rearm.triggered.connect(self.ctrl.router.rearm)
        for a in (self.act_exec, self.act_assist, act_replay):
            m.addAction(a)
        m.addSeparator()
        m.addAction(act_rearm)
        return m

    def _build_scene_tab(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        g = QGroupBox("Character / class inference")
        gl = QVBoxLayout(g)
        self.lbl_class = QLabel("Unknown")
        self.lbl_class.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self.lbl_class_ev = QLabel("")
        self.lbl_class_ev.setWordWrap(True)
        gl.addWidget(self.lbl_class)
        gl.addWidget(self.lbl_class_ev)
        l.addWidget(g)
        self.lbl_captcha = QLabel("CAPTCHA: not detected")
        l.addWidget(self.lbl_captcha)
        self.chk_overlay = QCheckBox("Show detection overlays")
        self.chk_overlay.setChecked(True)
        self.chk_overlay.toggled.connect(lambda on: setattr(self.preview, "show_overlay", on))
        l.addWidget(self.chk_overlay)
        self.txt_scene = QPlainTextEdit()
        self.txt_scene.setReadOnly(True)
        self.txt_scene.setMaximumHeight(130)
        l.addWidget(QLabel("Scene summary"))
        l.addWidget(self.txt_scene)
        self.tbl_entities = QTableWidget(0, 4)
        self.tbl_entities.setHorizontalHeaderLabels(["Label", "Conf.", "Track", "BBox"])
        self.tbl_entities.horizontalHeader().setStretchLastSection(True)
        self.tbl_entities.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        l.addWidget(QLabel("Detected entities / markers"))
        l.addWidget(self.tbl_entities, 1)
        return w

    def _build_action_tab(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        g = QGroupBox("Proposed next action (needs your approval)")
        gl = QVBoxLayout(g)
        self.lbl_prop = QLabel("— none —")
        self.lbl_prop.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.txt_why = QPlainTextEdit()
        self.txt_why.setReadOnly(True)
        self.txt_why.setMaximumHeight(110)
        row = QHBoxLayout()
        self.btn_approve = self._btn("✔ Approve && execute")
        self.btn_reject = self._btn("✘ Reject")
        self.btn_approve.clicked.connect(self.ctrl.approve_proposal)
        self.btn_reject.clicked.connect(lambda: self.ctrl.reject_proposal(None))
        row.addWidget(self.btn_approve)
        row.addWidget(self.btn_reject)
        row2 = QHBoxLayout()
        self.edt_intended = QLineEdit()
        self.edt_intended.setPlaceholderText("intended control, e.g. z / space / mouse_left")
        b = self._btn("Reject + provide intended action")
        b.clicked.connect(self.on_reject_with_intended)
        row2.addWidget(self.edt_intended)
        row2.addWidget(b)
        gl.addWidget(self.lbl_prop)
        gl.addWidget(self.txt_why)
        gl.addLayout(row)
        gl.addLayout(row2)
        l.addWidget(g)
        l.addWidget(QLabel("Recent observed / executed actions"))
        self.lst_actions = QListWidget()
        l.addWidget(self.lst_actions, 1)
        return w

    def _build_correction_tab(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        g = QGroupBox("Label a missed object")
        f = QFormLayout(g)
        self.chk_freeze = QCheckBox("Freeze preview (recommended before drawing)")
        self.chk_freeze.toggled.connect(self.on_freeze)
        self.cmb_label = QComboBox()
        self.cmb_label.setEditable(True)
        self.lbl_sel = QLabel("Draw a rectangle on the preview")
        self.chk_template = QCheckBox("Also save the crop as a detector template")
        self.chk_template.setChecked(True)
        b = self._btn("Save object label")
        b.clicked.connect(self.on_save_object)
        f.addRow(self.chk_freeze)
        f.addRow("Label:", self.cmb_label)
        f.addRow("Selection:", self.lbl_sel)
        f.addRow(self.chk_template)
        f.addRow(b)
        l.addWidget(g)
        g2 = QGroupBox("Annotate the current situation")
        l2 = QVBoxLayout(g2)
        self.txt_note = QPlainTextEdit()
        self.txt_note.setMaximumHeight(90)
        b2 = self._btn("Save annotation (goes to review)")
        b2.clicked.connect(self.on_annotate)
        l2.addWidget(self.txt_note)
        l2.addWidget(b2)
        l.addWidget(g2)
        l.addWidget(QLabel("All corrections become reviewable training data (Learning tab)."))
        l.addStretch(1)
        self.preview.selection_made.connect(
            lambda s: self.lbl_sel.setText(f"x={s[0]} y={s[1]} w={s[2]} h={s[3]}"))
        return w

    def _build_learning_tab(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        top = QHBoxLayout()
        self.cmb_filter = QComboBox()
        self.cmb_filter.addItems(["all", "pending", "approved", "rejected"])
        self.cmb_filter.currentTextChanged.connect(self.refresh_learning)
        top.addWidget(QLabel("Training examples:"))
        top.addWidget(self.cmb_filter)
        for text, fn in (("Approve", lambda: self.review("approved")),
                         ("Approve all", self.approve_all),
                         ("Reject", lambda: self.review("rejected")),
                         ("Reject all", self.reject_all),
                         ("Delete", self.delete_examples), ("Refresh", self.refresh_learning)):
            b = self._btn(text)
            b.clicked.connect(fn)
            top.addWidget(b)
        l.addLayout(top)
        self.tbl_ex = QTableWidget(0, 6)
        self.tbl_ex.setHorizontalHeaderLabels(["Id", "Kind", "Action", "Review", "Outcome", "Created"])
        self.tbl_ex.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl_ex.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tbl_ex.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tbl_ex.horizontalHeader().setStretchLastSection(True)
        l.addWidget(self.tbl_ex, 2)
        row = QHBoxLayout()
        row.addWidget(QLabel("Model versions (policy changes need explicit deployment):"))
        b1, b2 = self._btn("Train candidate model"), self._btn("Deploy selected version")
        b1.clicked.connect(self.on_train)
        b2.clicked.connect(self.on_deploy)
        row.addStretch(1)
        row.addWidget(b1)
        row.addWidget(b2)
        l.addLayout(row)
        self.tbl_models = QTableWidget(0, 5)
        self.tbl_models.setHorizontalHeaderLabels(["Ver.", "Status", "Examples", "Top-1 acc.", "Evaluation split"])
        self.tbl_models.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl_models.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tbl_models.horizontalHeader().setStretchLastSection(True)
        l.addWidget(self.tbl_models, 1)
        return w

    SETTING_FIELDS = [
        ("capture_fps", "int", "Capture FPS", (1, 60)),
        ("capture_scale", "float", "Capture scale", (0.25, 1.0)),
        ("record_video", "bool", "Record video (mp4)", None),
        ("save_frame_every_n", "int", "Store frame + detections every N frames", (1, 600)),
        ("recordings_dir", "str", "Recording folder", None),
        ("ocr_enabled", "bool", "Enable OCR (needs Tesseract)", None),
        ("tesseract_cmd", "str", "Tesseract executable (optional)", None),
        ("emergency_hotkey", "str", "Emergency stop hotkey (restart to apply)", None),
        ("max_actions_per_second", "int", "Max actions / second", (1, 60)),
        ("min_confidence", "float", "Min. confidence for proposals", (0.0, 1.0)),
        ("resume_delay_sec", "int", "Get-ready countdown after Resume (s)", (0, 60)),
        ("unknown_state_timeout_sec", "float", "Pause replay after unknown state (s)", (1.0, 120.0)),
        ("log_retention_days", "int", "Log retention (days)", (1, 3650)),
        ("db_backend", "choice", "Database backend (restart to apply)", ["tinydb", "mongodb"]),
        ("mongo_uri", "str", "MongoDB URI", None),
        ("tinydb_path", "str", "TinyDB file", None),
    ]

    def _build_settings_tab(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self._setting_widgets = {}
        for key, kind, label, extra in self.SETTING_FIELDS:
            val = self.ctrl.settings[key]
            if kind == "int":
                x = QSpinBox()
                x.setRange(*extra)
                x.setValue(int(val))
            elif kind == "float":
                x = QDoubleSpinBox()
                x.setRange(*extra)
                x.setSingleStep(0.05)
                x.setValue(float(val))
            elif kind == "bool":
                x = QCheckBox()
                x.setChecked(bool(val))
            elif kind == "choice":
                x = QComboBox()
                x.addItems(extra)
                x.setCurrentText(str(val))
            else:
                x = QLineEdit(str(val))
            self._setting_widgets[key] = (kind, x)
            f.addRow(label + ":", x)
        b = self._btn("Save settings")
        b.clicked.connect(self.on_save_settings)
        f.addRow(b)
        return w

    def _build_menu(self):
        mb = self.menuBar()
        pm = mb.addMenu("&Profile")
        for text, fn in (("New profile…", self.profile_new), ("Clone…", self.profile_clone),
                         ("Edit (JSON)…", self.profile_edit), ("Import…", self.profile_import),
                         ("Export…", self.profile_export), ("Delete", self.profile_delete)):
            a = QAction(text, self)
            a.triggered.connect(fn)
            pm.addAction(a)
        dm = mb.addMenu("&Data")
        for text, fn in (("Backup database", self.data_backup),
                         ("Export database to JSON…", self.data_export),
                         ("Delete a recorded session…", self.data_delete_session),
                         ("Purge old logs now", self.data_purge),
                         ("Open data folder", lambda: os.startfile(str(self.ctrl.settings.path("recordings_dir").parent))
                          if winutil.IS_WINDOWS else None)):
            a = QAction(text, self)
            a.triggered.connect(fn)
            dm.addAction(a)
        hm = mb.addMenu("&Help")
        a = QAction("Safety & limitations", self)
        a.triggered.connect(self.show_notice)
        hm.addAction(a)

    # ------------------------------------------------------------ handlers
    def show_notice(self, first_run=False) -> bool:
        txt = ("• No tool can guarantee 100% accurate simulation in every game. Check the "
               "per-game results and always keep the emergency stop (hotkey) at hand.\n"
               "• CAPTCHAs are only DETECTED: automation pauses and asks you to solve them. "
               "Nothing here solves or bypasses them.\n"
               "• Input injection may violate a game's terms of service or trigger anti-cheat. "
               "Use only with offline / automation-friendly games or with explicit permission.\n"
               "• Recordings and logs stay on this computer (data folder). You can delete or "
               "export them from the Data menu.\n"
               "• Nothing is sent to a game until you switch 'Input execution' on.")
        if not first_run:
            QMessageBox.information(self, "Safety & limitations", txt)
            return True
        r = QMessageBox.question(self, "Before you start", txt + "\n\nDo you accept?")
        return r == QMessageBox.StandardButton.Yes

    def refresh_windows(self):
        cur = self.cmb_window.currentData()
        self.cmb_window.blockSignals(True)
        self.cmb_window.clear()
        self.cmb_window.addItem("Primary monitor (full screen)", None)
        for w in self.ctrl.list_windows():
            self.cmb_window.addItem(f"{w['title'][:60]}  [{w['process']}]", w)
        self.cmb_window.blockSignals(False)
        if cur:
            for i in range(self.cmb_window.count()):
                d = self.cmb_window.itemData(i)
                if d and d["hwnd"] == cur["hwnd"]:
                    self.cmb_window.setCurrentIndex(i)
                    return
        else:
            auto = self.ctrl.find_window_for_profile()
            if auto:
                for i in range(self.cmb_window.count()):
                    d = self.cmb_window.itemData(i)
                    if d and d["hwnd"] == auto["hwnd"]:
                        self.cmb_window.setCurrentIndex(i)
                        return
        self.on_window_changed()

    def on_window_changed(self, *_):
        self.ctrl.select_target(self.cmb_window.currentData())

    def refresh_profiles(self):
        self.cmb_profile.blockSignals(True)
        self.cmb_profile.clear()
        for p in self.ctrl.profiles.list():
            self.cmb_profile.addItem(f"{p['name']}  (v{p.get('model_version', 0)})", p["_id"])
        i = self.cmb_profile.findData(self.ctrl.profile["_id"])
        self.cmb_profile.setCurrentIndex(max(0, i))
        self.cmb_profile.blockSignals(False)
        self.refresh_labels()

    def refresh_labels(self):
        cur = self.cmb_label.currentText()
        self.cmb_label.clear()
        self.cmb_label.addItems(self.ctrl.profile["detector"]["labels"])
        if cur:
            self.cmb_label.setCurrentText(cur)

    def on_profile_changed(self, *_):
        pid = self.cmb_profile.currentData()
        if pid and pid != self.ctrl.profile["_id"]:
            self.ctrl.select_profile(pid)
            self.refresh_labels()
            self.refresh_learning()

    def on_capture_toggled(self, on):
        if on:
            self.ctrl.start_capture()
        else:
            self.btn_record.setChecked(False)
            self.ctrl.stop_capture()

    def on_record_toggled(self, on):
        if on:
            if not self.ctrl.capture.running:
                QMessageBox.information(self, "Record", "Start 'Capture' first.")
                self.btn_record.blockSignals(True)
                self.btn_record.setChecked(False)
                self.btn_record.blockSignals(False)
                return
            self.ctrl.start_recording()
        else:
            self.ctrl.stop_recording()
            self.refresh_learning()

    def on_stop(self):
        self.ctrl.stop_all()
        for b in (self.btn_capture, self.btn_record):
            b.blockSignals(True)
            b.setChecked(False)
            b.blockSignals(False)
        self.act_exec.blockSignals(True)
        self.act_exec.setChecked(False)
        self.act_exec.blockSignals(False)
        self.refresh_learning()

    def on_exec_toggled(self, on):
        if on and not self._warned_execution:
            r = QMessageBox.warning(
                self, "Enable input execution",
                "The application will send keyboard/mouse/gamepad input to the selected "
                "window.\n\nThis may violate the game's terms of service or trigger anti-cheat. "
                "Use only where permitted. Press the emergency hotkey "
                f"({self.ctrl.settings['emergency_hotkey'].upper()}) or STOP to halt at any time.\n\n"
                "Enable?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if r != QMessageBox.StandardButton.Yes:
                self.act_exec.blockSignals(True)
                self.act_exec.setChecked(False)
                self.act_exec.blockSignals(False)
                return
            self._warned_execution = True
        if not self.ctrl.router.set_enabled(on) and on:
            self.act_exec.blockSignals(True)
            self.act_exec.setChecked(False)
            self.act_exec.blockSignals(False)

    def on_assist_toggled(self, on):
        if on and not self.ctrl.capture.running:
            QMessageBox.information(self, "Assist", "Start 'Capture' first.")
            self.act_assist.blockSignals(True)
            self.act_assist.setChecked(False)
            self.act_assist.blockSignals(False)
            return
        self.ctrl.set_assist(on)

    def on_replay(self):
        sessions = self.ctrl.list_sessions()
        if not sessions:
            QMessageBox.information(self, "Replay", "No completed recording sessions yet.")
            return
        items = []
        for s in sessions:
            when = datetime.fromtimestamp(s["start_time"]).strftime("%Y-%m-%d %H:%M")
            n = s.get("stats", {}).get("actions", "?")
            items.append(f"{when} | {s['application'].get('title', '')[:40]} | {n} actions | {s['_id'][:8]}")
        item, ok = QInputDialog.getItem(self, "Replay", "Session:", items, 0, False)
        if not ok:
            return
        sess = sessions[items.index(item)]
        speed, ok = QInputDialog.getDouble(self, "Replay", "Speed (1.0 = original):", 1.0, 0.25, 3.0, 2)
        if not ok:
            return
        r = QMessageBox.question(
            self, "Approve replay",
            f"Replay {sess.get('stats', {}).get('actions', '?')} recorded actions into:\n"
            f"{self.ctrl.target['title'] if self.ctrl.target else '(no window selected!)'}\n\n"
            "A 5 s countdown starts now - click into the game window. The replay pauses if the "
            "window loses focus, a CAPTCHA/unknown screen appears, or capture is lost.\n\nStart?")
        if r == QMessageBox.StandardButton.Yes:
            self.ctrl.replay_session(sess["_id"], speed)

    def on_reject_with_intended(self):
        self.ctrl.reject_proposal(self.edt_intended.text())
        self.edt_intended.clear()

    def on_freeze(self, on):
        self._frozen = self.ctrl.perception.latest() if on else None
        if on and self._frozen is not None:
            self.preview.set_frame(self._frozen.image, self._frozen.entities)

    def on_save_object(self):
        sel = self.preview.selection
        label = self.cmb_label.currentText().strip()
        if not sel or not label:
            QMessageBox.information(self, "Correction", "Draw a box on the preview and choose a label.")
            return
        try:
            self.ctrl.add_object_correction(label, sel, self.chk_template.isChecked(),
                                            res=self._frozen)
            self.preview.selection = None
            self.lbl_sel.setText("Saved. Draw another rectangle…")
        except Exception as exc:
            QMessageBox.warning(self, "Correction failed", str(exc))

    def on_annotate(self):
        text = self.txt_note.toPlainText().strip()
        if text:
            self.ctrl.annotate_situation(text)
            self.txt_note.clear()
            self.refresh_learning()

    def on_train(self):
        try:
            doc = self.ctrl.train()
        except Exception as exc:
            QMessageBox.warning(self, "Training", str(exc))
            return
        m = doc["evaluation_metrics"]
        QMessageBox.information(self, "Candidate model trained",
                                f"Version {doc['version']} (candidate, NOT active)\n"
                                f"Top-1 accuracy: {m.get('top1_accuracy')}\nSplit: {m['split']}\n"
                                f"Failures: {m.get('failure_categories')}\n\n"
                                "Select it and press 'Deploy' to make it the active policy.")
        self.refresh_learning()

    def on_deploy(self):
        rows = self.tbl_models.selectionModel().selectedRows()
        if not rows:
            QMessageBox.information(self, "Deploy", "Select a model version first.")
            return
        mid = self.tbl_models.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole)
        if QMessageBox.question(self, "Deploy", "Make this version the active policy for the "
                                "current profile?") == QMessageBox.StandardButton.Yes:
            self.ctrl.deploy(mid)
            self.refresh_profiles()
            self.refresh_learning()

    def on_save_settings(self):
        for key, (kind, w) in self._setting_widgets.items():
            if kind in ("int", "float"):
                self.ctrl.settings[key] = w.value()
            elif kind == "bool":
                self.ctrl.settings[key] = w.isChecked()
            elif kind == "choice":
                self.ctrl.settings[key] = w.currentText()
            else:
                self.ctrl.settings[key] = w.text().strip()
        self.ctrl.apply_settings()

    # ---- learning tab data
    def _ex_ids(self):
        return [self.tbl_ex.item(i.row(), 0).data(Qt.ItemDataRole.UserRole)
                for i in self.tbl_ex.selectionModel().selectedRows()]

    def review(self, status):
        ids = self._ex_ids()
        if ids:
            self.ctrl.set_review(ids, status)
            self.refresh_learning()

    def approve_all(self):
        n = len(self.ctrl.pending_trainable())
        if n == 0:
            QMessageBox.information(self, "Approve all", "No pending training examples.")
            return
        if QMessageBox.question(
                self, "Approve all",
                f"Approve all {n} pending training example(s) for this profile?\n\n"
                "Approved examples are used the next time you train a candidate model. "
                "The active policy does not change until you deploy a model.") == \
                QMessageBox.StandardButton.Yes:
            self.ctrl.approve_all_pending()
            self.refresh_learning()

    def reject_all(self):
        n = len(self.ctrl.pending_trainable())
        if n == 0:
            QMessageBox.information(self, "Reject all", "No pending training examples.")
            return
        if QMessageBox.question(
                self, "Reject all",
                f"Reject all {n} pending training example(s) for this profile?\n\n"
                "Rejected examples are excluded from training. They stay in the list "
                "until you delete them.") == QMessageBox.StandardButton.Yes:
            self.ctrl.reject_all_pending()
            self.refresh_learning()

    def delete_examples(self):
        ids = self._ex_ids()
        if ids and QMessageBox.question(self, "Delete", f"Delete {len(ids)} example(s)?") == \
                QMessageBox.StandardButton.Yes:
            self.ctrl.delete_examples(ids)
            self.refresh_learning()

    def refresh_learning(self, *_):
        flt = self.cmb_filter.currentText()
        exs = self.ctrl.list_examples(None if flt == "all" else flt)
        self.tbl_ex.setRowCount(len(exs))
        for r, e in enumerate(exs):
            act = e.get("action", {}).get("control", "—")
            if e.get("kind") == "rejection":
                act = f"✘ {e.get('rejected_action')}"
            if e.get("kind") == "annotation":
                act = (e.get("note", "")[:40])
            out = e.get("outcome")
            out_s = f"Δenemy {out['enemy_delta']:+d}" if out else "…"
            vals = [e["_id"][:8], e.get("kind", ""), act, e.get("review_status", ""), out_s,
                    e.get("created", "")]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(str(v))
                if c == 0:
                    it.setData(Qt.ItemDataRole.UserRole, e["_id"])
                self.tbl_ex.setItem(r, c, it)
        models = self.ctrl.list_models()
        self.tbl_models.setRowCount(len(models))
        for r, m in enumerate(models):
            ev = m.get("evaluation_metrics", {})
            vals = [f"v{m['version']}", m.get("status", ""), m["training_metadata"]["n_examples"],
                    ev.get("top1_accuracy"), ev.get("split", "")]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(str(v))
                if c == 0:
                    it.setData(Qt.ItemDataRole.UserRole, m["_id"])
                self.tbl_models.setItem(r, c, it)

    # ---- proposals / actions
    def show_proposal(self, p):
        if not p:
            self.lbl_prop.setText("— none —")
            self.txt_why.clear()
            self.btn_approve.setEnabled(False)
            self.btn_reject.setEnabled(False)
            return
        a = p["action"]
        self.lbl_prop.setText(f"{a['control']}  ({a['duration']}s)   confidence {a['confidence']:.2f}")
        self.txt_why.setPlainText("Why: nearest reviewed examples\n" + "\n".join(
            f"  • {x['control']:<10} dist {x['distance']:.3f} (state {x['state_distance']:.3f}, "
            f"sequence {x['sequence_distance']:.3f})" for x in a["explanation"]))
        self.btn_approve.setEnabled(True)
        self.btn_reject.setEnabled(True)
        self.tabs.setCurrentIndex(1)

    def add_action_row(self, a):
        t = datetime.fromtimestamp(a["timestamp"]).strftime("%H:%M:%S")
        self.lst_actions.insertItem(0, f"{t}  {a['input_device']:<8} {a['control']:<14} {a['duration']:.2f}s")
        while self.lst_actions.count() > 100:
            self.lst_actions.takeItem(self.lst_actions.count() - 1)

    # ---- profiles
    def profile_new(self):
        name, ok = QInputDialog.getText(self, "New profile", "Name:")
        if ok and name.strip():
            p = self.ctrl.profiles.create(name.strip())
            self.ctrl.select_profile(p["_id"])
            self.refresh_profiles()

    def profile_clone(self):
        name, ok = QInputDialog.getText(self, "Clone profile", "New name:",
                                        text=self.ctrl.profile["name"] + " copy")
        if ok and name.strip():
            p = self.ctrl.profiles.clone(self.ctrl.profile["_id"], name.strip())
            self.ctrl.select_profile(p["_id"])
            self.refresh_profiles()

    def profile_edit(self):
        p = dict(self.ctrl.profile)
        pid, created = p.pop("_id"), p.get("created")
        dlg = JsonDialog(self, f"Edit profile: {p['name']}", json.dumps(p, indent=2))
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            data = json.loads(dlg.edit.toPlainText())
            data["_id"], data["created"] = pid, created
            self.ctrl.profiles.save(data)
            self.ctrl.reload_profile()
            self.refresh_profiles()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid profile", str(exc))

    def profile_import(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import profile", "", "JSON (*.json)")
        if path:
            try:
                p = self.ctrl.profiles.import_json(path)
                self.ctrl.select_profile(p["_id"])
                self.refresh_profiles()
            except Exception as exc:
                QMessageBox.warning(self, "Import failed", str(exc))

    def profile_export(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export profile",
                                              f"{self.ctrl.profile['name']}.json", "JSON (*.json)")
        if path:
            self.ctrl.profiles.export_json(self.ctrl.profile["_id"], path)

    def profile_delete(self):
        if len(self.ctrl.profiles.list()) <= 1:
            QMessageBox.information(self, "Delete", "Keep at least one profile.")
            return
        if QMessageBox.question(self, "Delete", f"Delete profile '{self.ctrl.profile['name']}'?") == \
                QMessageBox.StandardButton.Yes:
            self.ctrl.profiles.delete(self.ctrl.profile["_id"])
            self.ctrl.select_profile(self.ctrl.profiles.list()[0]["_id"])
            self.refresh_profiles()

    # ---- data menu
    def data_backup(self):
        p = self.ctrl.db.backup()
        QMessageBox.information(self, "Backup", f"Saved to\n{p}")

    def data_export(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export database", "ai_player_sim_export.json",
                                              "JSON (*.json)")
        if path:
            self.ctrl.db.export_json(path)

    def data_delete_session(self):
        sessions = self.ctrl.db.find("sessions", sort=("start_time", -1))
        if not sessions:
            return
        items = [f"{datetime.fromtimestamp(s['start_time']).strftime('%Y-%m-%d %H:%M')} | "
                 f"{s.get('kind', '')} | {s['_id'][:8]}" for s in sessions]
        item, ok = QInputDialog.getItem(self, "Delete session", "Session (frames, video, actions "
                                        "and examples are removed):", items, 0, False)
        if ok:
            self.ctrl.db.delete_session(sessions[items.index(item)]["_id"])
            self.refresh_learning()

    def data_purge(self):
        n = self.ctrl.db.purge_logs(int(self.ctrl.settings["log_retention_days"]))
        QMessageBox.information(self, "Logs", f"Removed {n} old log records.")

    # ------------------------------------------------------------- timers
    def tick_preview(self):
        if self._frozen is not None:
            return
        f = self.ctrl.capture.latest()
        if f is None or f.index == self._last_frame_idx:
            return
        self._last_frame_idx = f.index
        res = self.ctrl.perception.latest()
        self.preview.set_frame(f.image, res.entities if res else [])

    def drain_logs(self):
        batch = []
        while self._log_queue and len(batch) < 500:
            batch.append(self._log_queue.popleft())
        if batch:
            self.logs.add_many(batch)

    def tick_info(self):
        res = self.ctrl.perception.latest()
        cap = self.ctrl.capture
        self.preview.capturing = cap.running and cap.ok
        if self.ctrl.resume_left > 0:
            self.preview.paused_text = (f"RESUMING IN {self.ctrl.resume_left} s — get ready "
                                        "(input is still blocked)")
        elif self.ctrl.router.paused_reason:
            self.preview.paused_text = (f"AUTOMATION PAUSED: {self.ctrl.router.paused_reason} — "
                                        "handle it, then press Resume")
        else:
            self.preview.paused_text = ""
        s = cap.stats
        self.lbl_status.setText(
            f"Capture {'OK' if cap.ok else ('LOST' if cap.running else 'idle')} | "
            f"{s['fps']:.1f} fps | frames {s['frames']} | drops {s['drops']} | "
            f"failures {s['failures']} | perception {self.ctrl.perception.avg_ms:.0f} ms | "
            f"policy {'v' + str(self.ctrl.profile.get('model_version')) if self.ctrl.policy else 'none'}")
        if res is None:
            return
        ci = res.class_info
        self.lbl_class.setText(f"{ci['candidate']}   ({ci['confidence']:.0%})")
        self.lbl_class_ev.setText("Evidence: " + (", ".join(ci["evidence"]) or "none"))
        cp = res.captcha
        self.lbl_captcha.setText("CAPTCHA: DETECTED — " + cp.get("reason", "") if cp.get("detected")
                                 else "CAPTCHA: not detected")
        sc = res.scene
        self.txt_scene.setPlainText(
            f"Scene: {sc['scene_type']}\n"
            + ", ".join(f"{k}={v}" for k, v in sc["counts"].items() if v)
            + ("\nNearest enemy: dx={:.2f} dy={:.2f}".format(sc["nearest_enemy"]["dx"],
                                                             sc["nearest_enemy"]["dy"])
               if sc.get("nearest_enemy") else "")
            + ("\nOCR: " + " | ".join(sc["ocr_text"][:5]) if sc["ocr_text"] else ""))
        ents = res.entities[:60]
        self.tbl_entities.setRowCount(len(ents))
        for r, e in enumerate(ents):
            for c, v in enumerate([e.label, f"{e.confidence:.2f}", e.track_id, list(e.bbox)]):
                self.tbl_entities.setItem(r, c, QTableWidgetItem(str(v)))

    def refresh_state(self):
        c = self.ctrl
        for b, val in ((self.btn_capture, c.capture.running), (self.btn_record, c.session is not None),
                       (self.btn_learn, c.learn_enabled)):
            b.blockSignals(True)
            b.setChecked(val)
            b.blockSignals(False)
        for a, val in ((self.act_exec, c.router.enabled), (self.act_assist, c.assist_enabled)):
            a.blockSignals(True)
            a.setChecked(val)
            a.blockSignals(False)
        text = c.router.state_text()
        key = "EMERGENCY" if text.startswith("EMERGENCY") else "PAUSED" if text.startswith("PAUSED") \
            else text
        suffix = " · REPLAYING" if c.replay_active else ""
        self.lbl_auto.setText(f"AUTOMATION: {text}{suffix}")
        self.lbl_auto.setStyleSheet(STATE_STYLE.get(key, "") + ";padding:6px;border-radius:4px")
        counting = c.resume_left > 0
        self.btn_resume.setEnabled(bool(c.router.paused_reason) and not counting)
        self.btn_resume.setText(f"Resuming in {c.resume_left}…" if counting else "Resume")
        self.btn_approve.setEnabled(c.proposal is not None)
        self.btn_reject.setEnabled(c.proposal is not None)

    def closeEvent(self, e):
        self.ctrl.shutdown()
        e.accept()
