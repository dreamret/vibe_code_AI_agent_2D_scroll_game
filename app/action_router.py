"""Action router: unified input adapters + safety checks.

Safety gates, in order, before any input is sent:
  emergency stop -> human enable switch -> pause (CAPTCHA/unknown/...) ->
  profile permission -> rate limit -> target-window focus verification.
"""
from __future__ import annotations

import threading
import time
from collections import deque

from . import winutil


def parse_key(name: str):
    from pynput.keyboard import Key
    name = name.strip()
    if len(name) == 1:
        return name
    if hasattr(Key, name):
        return getattr(Key, name)
    raise ValueError(f"Unknown key '{name}'")


class KeyboardAdapter:
    available = True

    def __init__(self):
        from pynput.keyboard import Controller
        self.kb = Controller()
        self.held: set = set()

    def execute(self, action, window):
        key = parse_key(action["control"])
        self.kb.press(key)
        self.held.add(key)
        try:
            time.sleep(max(0.02, min(float(action.get("duration", 0.05)), 10.0)))
        finally:
            self.kb.release(key)
            self.held.discard(key)

    def release_all(self):
        for k in list(self.held):
            try:
                self.kb.release(k)
            except Exception:
                pass
        self.held.clear()


class MouseAdapter:
    available = True

    def __init__(self):
        from pynput.mouse import Button, Controller
        self.ms = Controller()
        self.Button = Button

    def execute(self, action, window):
        btn = getattr(self.Button, action["control"].replace("mouse_", ""), None)
        if btn is None:
            raise ValueError(f"Unknown mouse control {action['control']}")
        if window and "x" in action:
            r = winutil.client_rect(window["hwnd"])
            if r:
                self.ms.position = (r["left"] + int(action["x"]), r["top"] + int(action["y"]))
        self.ms.press(btn)
        time.sleep(max(0.02, min(float(action.get("duration", 0.05)), 5.0)))
        self.ms.release(btn)

    def release_all(self):
        pass


class GamepadAdapter:
    """Virtual Xbox 360 pad via the optional `vgamepad` package (needs ViGEmBus)."""
    BUTTONS = ["A", "B", "X", "Y", "LEFT_SHOULDER", "RIGHT_SHOULDER", "BACK", "START",
               "LEFT_THUMB", "RIGHT_THUMB"]

    def __init__(self):
        self.available = False
        self.reason = ""
        try:
            import vgamepad as vg
            self.vg = vg
            self.pad = vg.VX360Gamepad()
            self.available = True
        except Exception as exc:
            self.reason = f"vgamepad not usable ({exc.__class__.__name__})"

    def execute(self, action, window):
        if not self.available:
            raise RuntimeError(self.reason)
        c = action["control"]
        if not c.startswith("gamepad_button_"):
            raise RuntimeError(f"Unsupported gamepad control {c}")
        idx = int(c.rsplit("_", 1)[1])
        if idx >= len(self.BUTTONS):
            raise RuntimeError(f"No mapping for button {idx}")
        btn = getattr(self.vg.XUSB_BUTTON, f"XUSB_GAMEPAD_{self.BUTTONS[idx]}")
        self.pad.press_button(button=btn)
        self.pad.update()
        time.sleep(max(0.02, min(float(action.get("duration", 0.05)), 5.0)))
        self.pad.release_button(button=btn)
        self.pad.update()

    def release_all(self):
        if self.available:
            self.pad.reset()
            self.pad.update()


class TouchAdapter:
    """Touch injection needs an OS/device pathway that is not available generically."""
    available = False

    def execute(self, action, window):
        raise RuntimeError("Touch injection is not supported on this system/profile")

    def release_all(self):
        pass


class ActionRouter:
    def __init__(self, settings, log, get_target, get_profile):
        self.settings = settings
        self.log = log
        self.get_target = get_target
        self.get_profile = get_profile
        self.enabled = False                # human-controlled enable switch
        self.emergency = False
        self.paused_reason = ""
        self.listeners = []                 # callbacks(state_str)
        self._times: deque = deque()
        self._lock = threading.Lock()
        self._adapters: dict = {}
        self._hotkey = None

    # ---- adapters (lazy) ----------------------------------------------
    def _adapter(self, device):
        if device not in self._adapters:
            cls = {"keyboard": KeyboardAdapter, "mouse": MouseAdapter,
                   "gamepad": GamepadAdapter, "touch": TouchAdapter}.get(device)
            if cls is None:
                raise ValueError(f"Unknown input device '{device}'")
            self._adapters[device] = cls()
        return self._adapters[device]

    # ---- state ---------------------------------------------------------
    def state_text(self) -> str:
        if self.emergency:
            return "EMERGENCY STOP"
        if not self.enabled:
            return "OFF"
        if self.paused_reason:
            return f"PAUSED ({self.paused_reason})"
        return "ARMED"

    def _notify(self):
        for cb in list(self.listeners):
            try:
                cb(self.state_text())
            except Exception:
                pass

    def set_enabled(self, on: bool):
        if on and self.emergency:
            self.log.warning("router", "Re-arm the emergency stop first", event_id="ROUTER_BLOCKED")
            return False
        self.enabled = on
        self.log.info("router", f"Input execution {'ENABLED' if on else 'disabled'} by user",
                      event_id="ROUTER_ENABLE" if on else "ROUTER_DISABLE")
        self._notify()
        return True

    def pause(self, reason: str):
        if self.paused_reason != reason:
            self.paused_reason = reason
            self.log.warning("router", f"Automation PAUSED: {reason}. Human review required.",
                             event_id="ROUTER_PAUSE", reason=reason)
            self._notify()

    def resume(self):
        if self.paused_reason:
            self.log.info("router", "Automation resumed by user", event_id="ROUTER_RESUME")
        self.paused_reason = ""
        self._notify()

    def emergency_stop(self):
        self.emergency = True
        self.enabled = False
        for a in self._adapters.values():
            try:
                a.release_all()
            except Exception:
                pass
        self.log.critical("router", "EMERGENCY STOP - all input execution halted",
                          event_id="EMERGENCY_STOP")
        self._notify()

    def rearm(self):
        self.emergency = False
        self.log.info("router", "Emergency stop cleared (automation remains OFF)",
                      event_id="EMERGENCY_CLEAR")
        self._notify()

    # ---- global emergency hotkey --------------------------------------
    def start_hotkey(self):
        try:
            from pynput import keyboard
            name = self.settings["emergency_hotkey"].strip().lower()
            combo = name if (name.startswith("<") or "+" in name) else f"<{name}>"
            self._hotkey = keyboard.GlobalHotKeys({combo: self.emergency_stop})
            self._hotkey.start()
            self.log.info("router", f"Emergency hotkey active: {name.upper()}",
                          event_id="HOTKEY_READY")
        except Exception as exc:
            self.log.error("router", f"Emergency hotkey unavailable: {exc}",
                           event_id="HOTKEY_FAIL")

    def stop_hotkey(self):
        if self._hotkey:
            self._hotkey.stop()

    # ---- checks --------------------------------------------------------
    def verify_focus(self) -> bool:
        t = self.get_target()
        if not t or not t.get("hwnd"):
            return False                    # never inject into an unidentified target
        return winutil.is_window(t["hwnd"]) and winutil.foreground_hwnd() == t["hwnd"]

    def _permitted(self, control: str) -> bool:
        prof = self.get_profile() or {}
        allowed = prof.get("constraints", {}).get("allowed_controls") or \
            list(prof.get("input_mapping", {}).values())
        return control in allowed

    def _rate_ok(self) -> bool:
        prof = self.get_profile() or {}
        limit = min(int(self.settings["max_actions_per_second"]),
                    int(prof.get("constraints", {}).get("max_actions_per_second", 10)))
        now = time.time()
        with self._lock:
            while self._times and now - self._times[0] > 1.0:
                self._times.popleft()
            if len(self._times) >= max(1, limit):
                return False
            self._times.append(now)
            return True

    # ---- execute -------------------------------------------------------
    def execute(self, action: dict, origin: str = "replay") -> dict:
        """Returns {'ok': bool, 'reason': str, 'latency_ms': float, 'delivered_at': float}"""
        req = time.time()

        def fail(reason, level="warning"):
            getattr(self.log, level)("router", f"Action rejected: {reason}",
                                     event_id="ACTION_REJECTED", control=action.get("control"),
                                     reason=reason, origin=origin)
            return {"ok": False, "reason": reason, "requested_at": req}

        if self.emergency:
            return fail("emergency_stop")
        if not self.enabled:
            return fail("automation_disabled")
        if self.paused_reason:
            return fail(f"paused:{self.paused_reason}")
        if not self._permitted(action.get("control", "")):
            return fail("control_not_permitted_by_profile")
        if not self._rate_ok():
            return fail("rate_limit")
        if not self.verify_focus():
            return fail("target_window_not_focused")
        try:
            adapter = self._adapter(action.get("input_device", "keyboard"))
            if not adapter.available:
                return fail(f"adapter_unavailable:{action.get('input_device')}")
            sent = time.time()                    # moment the press is injected
            adapter.execute(action, self.get_target())
        except Exception as exc:
            return fail(f"adapter_error:{exc}", "error")
        self.log.debug("router", f"Executed {action['control']}", event_id="ACTION_EXECUTED",
                       origin=origin, duration=action.get("duration"))
        return {"ok": True, "reason": "", "requested_at": req, "delivered_at": sent,
                "latency_ms": round((sent - req) * 1000, 2)}
