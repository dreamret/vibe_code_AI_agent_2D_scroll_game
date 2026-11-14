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


# ---- key names understood by the window-message adapter (same names the recorder writes)
VK = {"space": 0x20, "enter": 0x0D, "esc": 0x1B, "tab": 0x09, "backspace": 0x08,
      "shift": 0x10, "shift_l": 0xA0, "shift_r": 0xA1, "ctrl": 0x11, "ctrl_l": 0xA2,
      "ctrl_r": 0xA3, "alt": 0x12, "alt_l": 0xA4, "alt_r": 0xA5, "alt_gr": 0xA5,
      "caps_lock": 0x14, "num_lock": 0x90, "insert": 0x2D, "delete": 0x2E, "home": 0x24,
      "end": 0x23, "page_up": 0x21, "page_down": 0x22, "left": 0x25, "up": 0x26,
      "right": 0x27, "down": 0x28}
VK.update({f"f{i}": 0x6F + i for i in range(1, 25)})
EXTENDED = {0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x2D, 0x2E, 0xA3, 0xA5}
SYS_KEYS = {0x12, 0xA4, 0xA5}
MOUSE_BUTTONS = {"mouse_left": (0x0201, 0x0202, 0x0001), "mouse_right": (0x0204, 0x0205, 0x0002),
                 "mouse_middle": (0x0207, 0x0208, 0x0010)}


def validate_control(control: str) -> str:
    """Normalises and checks a key/button name. Raises ValueError with a helpful message."""
    c = (control or "").strip()
    if not c:
        raise ValueError("Key/button is empty")
    low = c.lower()
    if low.startswith("mouse_"):
        if low not in MOUSE_BUTTONS:
            raise ValueError("Mouse buttons: mouse_left, mouse_right, mouse_middle")
        return low
    if low.startswith("gamepad_button_"):
        if not low.rsplit("_", 1)[1].isdigit():
            raise ValueError("Gamepad buttons look like gamepad_button_0")
        return low
    if len(c) == 1:
        return c.lower()
    if low in VK:
        return low
    raise ValueError(f"Unknown key name '{c}'. Examples: z, space, left, enter, shift, f5, mouse_left")


def vk_for(name: str) -> int:
    if name in VK:
        return VK[name]
    if len(name) == 1:
        if name.isalnum() and name.isascii():
            return ord(name.upper())
        v = winutil.vk_from_char(name)
        if v:
            return v
    raise ValueError(f"No virtual-key code for '{name}'")


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


class UnsupportedAdapter:
    available = False

    def __init__(self, reason):
        self.reason = reason

    def execute(self, action, window):
        raise RuntimeError(self.reason)

    def release_all(self):
        pass


def _target_hwnd(window, children: bool) -> tuple[int, int]:
    if not window or not window.get("hwnd"):
        raise RuntimeError("no target window selected")
    top = window["hwnd"]
    if not winutil.IS_WINDOWS or not winutil.is_window(top):
        raise RuntimeError("target window is gone")
    return top, (winutil.deepest_child(top) if children else top)


class WinMsgKeyboardAdapter:
    """Sends WM_KEYDOWN/WM_KEYUP straight to the game window (no OS keyboard input,
    no focus needed). Works for games that read window messages; games that read
    raw input / DirectInput / GetAsyncKeyState ignore it - use input_mode=os_input then."""
    available = True

    def __init__(self, settings):
        self.settings = settings
        self.held: set = set()
        self.reason = "" if winutil.IS_WINDOWS else "window messages need Windows"
        self.available = winutil.IS_WINDOWS

    def _lparams(self, vk):
        scan = winutil.map_vk_to_scan(vk)
        ext = 1 if vk in EXTENDED else 0
        down = 1 | (scan << 16) | (ext << 24)
        return down, down | (1 << 30) | (1 << 31)

    def _post(self, hwnd, vk, down: bool):
        d, u = self._lparams(vk)
        sysk = vk in SYS_KEYS
        msg = (0x104 if down else 0x105) if sysk else (0x100 if down else 0x101)
        if not winutil.post_message(hwnd, msg, vk, d if down else u):
            raise RuntimeError("PostMessage failed (is the game running as administrator?)")

    def execute(self, action, window):
        _top, hwnd = _target_hwnd(window, bool(self.settings.get("post_to_child_window", True)))
        vk = vk_for(action["control"])
        self._post(hwnd, vk, True)
        self.held.add((hwnd, vk))
        try:
            time.sleep(max(0.02, min(float(action.get("duration", 0.05)), 10.0)))
        finally:
            self._post(hwnd, vk, False)
            self.held.discard((hwnd, vk))

    def release_all(self):
        for hwnd, vk in list(self.held):
            try:
                self._post(hwnd, vk, False)
            except Exception:
                pass
        self.held.clear()


class WinMsgMouseAdapter:
    """WM_MOUSEMOVE + WM_xBUTTONDOWN/UP sent to the game window (client coordinates)."""

    def __init__(self, settings):
        self.settings = settings
        self.reason = "" if winutil.IS_WINDOWS else "window messages need Windows"
        self.available = winutil.IS_WINDOWS

    def execute(self, action, window):
        top, hwnd = _target_hwnd(window, bool(self.settings.get("post_to_child_window", True)))
        spec = MOUSE_BUTTONS.get(action["control"])
        if not spec:
            raise RuntimeError(f"Unknown mouse control {action['control']}")
        down, up, mk = spec
        x, y = winutil.client_point(hwnd, top, int(action.get("x", 0)), int(action.get("y", 0)))
        lp = ((y & 0xFFFF) << 16) | (x & 0xFFFF)
        winutil.post_message(hwnd, 0x0200, 0, lp)               # WM_MOUSEMOVE
        winutil.post_message(hwnd, down, mk, lp)
        time.sleep(max(0.02, min(float(action.get("duration", 0.05)), 5.0)))
        winutil.post_message(hwnd, up, 0, lp)

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
    @property
    def mode(self) -> str:
        return str(self.settings.get("input_mode", "window_message"))

    def _adapter(self, device):
        key = (self.mode, device)
        if key not in self._adapters:
            if self.mode == "window_message":
                table = {"keyboard": lambda: WinMsgKeyboardAdapter(self.settings),
                         "mouse": lambda: WinMsgMouseAdapter(self.settings),
                         "gamepad": lambda: UnsupportedAdapter(
                             "gamepad can't be sent as window messages - use input_mode=os_input"),
                         "touch": TouchAdapter}
            else:
                table = {"keyboard": KeyboardAdapter, "mouse": MouseAdapter,
                         "gamepad": GamepadAdapter, "touch": TouchAdapter}
            if device not in table:
                raise ValueError(f"Unknown input device '{device}'")
            self._adapters[key] = table[device]()
        return self._adapters[key]

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
        """os_input: the game must be the foreground window (the OS routes keys there).
        window_message: input goes to that one window only, so it just has to exist."""
        t = self.get_target()
        if not t or not t.get("hwnd"):
            return False                    # never inject into an unidentified target
        if not winutil.is_window(t["hwnd"]):
            return False
        if self.mode == "window_message":
            return not winutil.is_minimized(t["hwnd"])
        return winutil.foreground_hwnd() == t["hwnd"]

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
            return fail("target_window_missing" if self.mode == "window_message"
                        else "target_window_not_focused")
        try:
            adapter = self._adapter(action.get("input_device", "keyboard"))
            if not adapter.available:
                return fail(f"adapter_unavailable:{action.get('input_device')}:"
                            f"{getattr(adapter, 'reason', '')}")
            sent = time.time()                    # moment the press is injected
            adapter.execute(action, self.get_target())
        except Exception as exc:
            return fail(f"adapter_error:{exc}", "error")
        self.log.debug("router", f"Executed {action['control']}", event_id="ACTION_EXECUTED",
                       origin=origin, duration=action.get("duration"), mode=self.mode)
        return {"ok": True, "reason": "", "requested_at": req, "delivered_at": sent,
                "latency_ms": round((sent - req) * 1000, 2)}
