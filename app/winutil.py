"""Windows helpers: list windows, client rectangles, foreground check."""
from __future__ import annotations

import ctypes
import sys

IS_WINDOWS = sys.platform.startswith("win")
OWN_TITLES = ("Local AI Player-Action Simulator",)
_IGNORED = {"Program Manager", "Windows Input Experience", "Microsoft Text Input Application"}

if IS_WINDOWS:
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    H = wintypes.HWND
    user32.IsWindow.argtypes = [H]
    user32.IsWindowVisible.argtypes = [H]
    user32.IsIconic.argtypes = [H]
    user32.GetWindowTextLengthW.argtypes = [H]
    user32.GetWindowTextW.argtypes = [H, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowThreadProcessId.argtypes = [H, ctypes.POINTER(wintypes.DWORD)]
    user32.GetClientRect.argtypes = [H, ctypes.POINTER(wintypes.RECT)]
    user32.ClientToScreen.argtypes = [H, ctypes.POINTER(wintypes.POINT)]
    user32.GetForegroundWindow.restype = wintypes.HWND
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]


def _process_name(pid: int) -> str:
    if not IS_WINDOWS:
        return ""
    h = kernel32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(1024)
        if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            return buf.value.replace("\\", "/").split("/")[-1]
    finally:
        kernel32.CloseHandle(h)
    return ""


def list_windows() -> list[dict]:
    """Visible, titled, non-minimised top-level windows."""
    results: list[dict] = []
    if not IS_WINDOWS:
        return results
    proc_t = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def cb(hwnd, _lparam):
        try:
            if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
                return True
            n = user32.GetWindowTextLengthW(hwnd)
            if n <= 0:
                return True
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            title = buf.value.strip()
            if not title or title in _IGNORED or title in OWN_TITLES:
                return True
            pid = wintypes.DWORD(0)
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            rect = client_rect(int(hwnd))
            if rect and rect["width"] > 80 and rect["height"] > 60:
                results.append({"hwnd": int(hwnd), "title": title, "pid": pid.value,
                                "process": _process_name(pid.value)})
        except Exception:
            pass
        return True

    user32.EnumWindows(proc_t(cb), 0)
    return results


def is_window(hwnd: int) -> bool:
    return bool(IS_WINDOWS and hwnd and user32.IsWindow(hwnd))


def client_rect(hwnd: int) -> dict | None:
    """Client area in screen coordinates: left, top, width, height."""
    if not IS_WINDOWS or not hwnd:
        return None
    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None
    pt = wintypes.POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(pt)):
        return None
    return {"left": int(pt.x), "top": int(pt.y),
            "width": int(rect.right - rect.left), "height": int(rect.bottom - rect.top)}


def foreground_hwnd() -> int:
    if not IS_WINDOWS:
        return 0
    return int(user32.GetForegroundWindow() or 0)


def window_title(hwnd: int) -> str:
    if not IS_WINDOWS or not hwnd:
        return ""
    n = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value
