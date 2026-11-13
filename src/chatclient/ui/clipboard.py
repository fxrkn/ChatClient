"""Copy text to the system clipboard on Windows, macOS and Linux.

Order: native API/tools first; if none work (e.g. a Linux server over SSH), fall back
to the OSC 52 escape sequence, which most modern terminals turn into a clipboard copy.
"""

from __future__ import annotations

import base64
import os
import shutil
import subprocess
import sys

LINUX_TOOLS = (
    ["wl-copy"],
    ["xclip", "-selection", "clipboard"],
    ["xsel", "--clipboard", "--input"],
)


def copy_text(text: str) -> str:
    """Copy ``text``; returns how it was done ("system" or "terminal")."""
    try:
        if os.name == "nt":
            _windows_copy(text)
            return "system"
        if sys.platform == "darwin" and shutil.which("pbcopy"):
            subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True, timeout=5)
            return "system"
        for cmd in LINUX_TOOLS:
            if shutil.which(cmd[0]):
                subprocess.run(cmd, input=text.encode("utf-8"), check=True, timeout=5)
                return "system"
    except (OSError, subprocess.SubprocessError):
        pass
    _osc52_copy(text)
    return "terminal"


def _osc52_copy(text: str) -> None:
    payload = base64.b64encode(text.encode("utf-8")).decode("ascii")
    sys.stdout.write(f"\x1b]52;c;{payload}\x07")
    sys.stdout.flush()


def _windows_api():  # pragma: no cover - Windows only
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.OpenClipboard.restype = wintypes.BOOL
    user32.EmptyClipboard.restype = wintypes.BOOL
    user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
    user32.SetClipboardData.restype = wintypes.HANDLE
    user32.GetClipboardData.argtypes = [wintypes.UINT]
    user32.GetClipboardData.restype = wintypes.HANDLE
    user32.CloseClipboard.restype = wintypes.BOOL
    kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalLock.restype = wintypes.LPVOID
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalFree.argtypes = [wintypes.HGLOBAL]
    return ctypes, user32, kernel32


CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002


def _windows_copy(text: str) -> None:  # pragma: no cover - Windows only
    ctypes, user32, kernel32 = _windows_api()
    data = text.encode("utf-16-le") + b"\x00\x00"
    if not user32.OpenClipboard(None):
        raise OSError("could not open the clipboard")
    try:
        user32.EmptyClipboard()
        handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data))
        if not handle:
            raise OSError("GlobalAlloc failed")
        ptr = kernel32.GlobalLock(handle)
        ctypes.memmove(ptr, data, len(data))
        kernel32.GlobalUnlock(handle)
        if not user32.SetClipboardData(CF_UNICODETEXT, handle):
            kernel32.GlobalFree(handle)
            raise OSError("SetClipboardData failed")
    finally:
        user32.CloseClipboard()


def read_text_windows() -> str | None:  # pragma: no cover - used by tests on Windows
    ctypes, user32, kernel32 = _windows_api()
    if not user32.OpenClipboard(None):
        return None
    try:
        handle = user32.GetClipboardData(CF_UNICODETEXT)
        if not handle:
            return None
        ptr = kernel32.GlobalLock(handle)
        try:
            return ctypes.wstring_at(ptr)
        finally:
            kernel32.GlobalUnlock(handle)
    finally:
        user32.CloseClipboard()
