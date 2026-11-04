"""Input recorder: keyboard / mouse / gamepad events -> normalised action dicts.

Only events that happen while the selected game window is in the foreground
are stored (privacy: minimise collection). Touch input cannot be observed by
this process; it is listed as unsupported.
"""
from __future__ import annotations

import os
import threading
import time

from . import winutil


def key_name(key) -> str:
    ch = getattr(key, "char", None)
    if ch and ord(ch) >= 32:
        return ch.lower()
    name = getattr(key, "name", None)
    if name:
        return name
    vk = getattr(key, "vk", None)
    return f"vk_{vk}" if vk is not None else str(key)


class InputRecorder:
    def __init__(self, settings, log, get_target, on_action):
        self.settings = settings
        self.log = log
        self.get_target = get_target          # () -> target dict | None
        self.on_action = on_action            # (action dict) -> None
        self._kb = None
        self._ms = None
        self._pad_thread = None
        self._stop = threading.Event()
        self._down: dict[str, float] = {}
        self.running = False

    # ---- helpers -------------------------------------------------------
    def _focused(self) -> bool:
        t = self.get_target()
        if not t or not t.get("hwnd"):
            return True                       # whole-screen mode
        return winutil.foreground_hwnd() == t["hwnd"]

    def _rel(self, x, y):
        t = self.get_target()
        if t and t.get("hwnd"):
            r = winutil.client_rect(t["hwnd"])
            if r:
                return x - r["left"], y - r["top"]
        return x, y

    def _emit(self, device, control, ts, duration, **extra):
        t = self.get_target()
        act = {"timestamp": ts, "input_device": device, "control": control,
               "value": "press", "duration": round(duration, 4),
               "target_window": t["title"] if t else "screen", "origin": "human"}
        act.update(extra)
        try:
            self.on_action(act)
        except Exception as exc:
            self.log.error("input", f"Action handler failed: {exc}")

    # ---- keyboard / mouse ---------------------------------------------
    def _on_press(self, key):
        name = key_name(key)
        if name == self.settings["emergency_hotkey"].lower() or name in self._down:
            return
        if self._focused():
            self._down[name] = time.time()

    def _on_release(self, key):
        name = key_name(key)
        t0 = self._down.pop(name, None)
        if t0 is not None:
            self._emit("keyboard", name, t0, time.time() - t0)

    def _on_click(self, x, y, button, pressed):
        name = f"mouse_{button.name}"
        if pressed:
            if self._focused():
                rx, ry = self._rel(x, y)
                self._down[name] = (time.time(), rx, ry)
        else:
            d = self._down.pop(name, None)
            if d:
                self._emit("mouse", name, d[0], time.time() - d[0], x=int(d[1]), y=int(d[2]))

    # ---- gamepad (optional: pygame) -----------------------------------
    def _pad_loop(self):
        try:
            os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
            import pygame
        except ImportError:
            self.log.info("input", "pygame not installed - gamepad recording disabled",
                          event_id="GAMEPAD_DISABLED")
            return
        pygame.init()
        pygame.joystick.init()
        js = None
        down: dict[str, float] = {}
        announced = False
        while not self._stop.is_set():
            pygame.event.pump()
            if js is None:
                pygame.joystick.quit()
                pygame.joystick.init()
                if pygame.joystick.get_count():
                    js = pygame.joystick.Joystick(0)
                    js.init()
                    self.log.info("input", f"Gamepad detected: {js.get_name()}",
                                  event_id="GAMEPAD_FOUND")
                else:
                    if not announced:
                        self.log.info("input", "No gamepad connected", event_id="GAMEPAD_NONE")
                        announced = True
                    time.sleep(1.0)
                    continue
            try:
                states = {f"gamepad_button_{i}": js.get_button(i) for i in range(js.get_numbuttons())}
                for a in range(js.get_numaxes()):
                    v = js.get_axis(a)
                    states[f"gamepad_axis_{a}_pos"] = v > 0.6
                    states[f"gamepad_axis_{a}_neg"] = v < -0.6
            except Exception:
                js = None
                continue
            focused = self._focused()
            for name, on in states.items():
                if on and name not in down and focused:
                    down[name] = time.time()
                elif not on and name in down:
                    t0 = down.pop(name)
                    self._emit("gamepad", name, t0, time.time() - t0)
            time.sleep(1 / 120)
        pygame.joystick.quit()

    # ---- control -------------------------------------------------------
    def start(self):
        if self.running:
            return
        self._stop.clear()
        self._down.clear()
        try:
            from pynput import keyboard, mouse
            self._kb = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
            self._ms = mouse.Listener(on_click=self._on_click)
            self._kb.start()
            self._ms.start()
        except Exception as exc:
            self.log.error("input", f"Cannot start keyboard/mouse listeners: {exc}",
                           event_id="INPUT_FAIL")
            return
        self._pad_thread = threading.Thread(target=self._pad_loop, daemon=True, name="gamepad")
        self._pad_thread.start()
        self.running = True
        self.log.info("input", "Input recording started (events only while the target "
                      "window is focused). Touch input is not recordable.",
                      event_id="INPUT_START")

    def stop(self):
        if not self.running:
            return
        self._stop.set()
        for lst in (self._kb, self._ms):
            try:
                lst.stop()
            except Exception:
                pass
        self.running = False
        self.log.info("input", "Input recording stopped", event_id="INPUT_STOP")
