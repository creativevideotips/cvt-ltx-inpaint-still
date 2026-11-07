"""Everything in the Python side that is specific to macOS (decision 64). CVT port (cvt1.17): Mac
only; upstream's Linux branches (zenity windows, /proc) are gone. Standard library only
(setup_helper.py and job.py run under the system python3).

The progress window is mac_progress.swift, a small AppKit program that reads zenity's stdin
protocol, compiled once with swiftc (Command Line Tools, which Setup requires) into
~/.cache/ltx-inpaint-still/bin. Chosen over the alternatives on a stock macOS 15: AppleScript's
"progress" properties draw nothing when run by osascript (only in applets/Script Editor); the
system python3's tkinter (Tk 8.5) aborts at Tk() ("macOS 15 (1507) or later required"); a JXA
(osascript -l JavaScript) Cocoa window works but needs the untyped ObjC bridge for the button
target and the stdin reader; `swift script.swift` would recompile on every job. Messages are
osascript `display dialog`; ps gives the process name.
"""

from __future__ import annotations

import hashlib
import os
import subprocess


class ProgressWindow:
    """A progress window with a Cancel button. Without a UI every call is a no-op."""

    def __init__(self, title: str, width: int, enabled: bool) -> None:
        self.proc = None
        helper = _mac_progress_helper() if enabled else ""
        if helper:  # zenity's stdin protocol; exits 1 on Cancel or when closed
            self.proc = subprocess.Popen([helper, title, str(width)], stdin=subprocess.PIPE,
                                         text=True, encoding="utf-8")

    @property
    def shown(self) -> bool:
        return self.proc is not None

    def gone(self) -> bool:
        """True once the window has closed (the user pressed Cancel, or it was closed)."""
        return bool(self.proc) and self.proc.poll() is not None

    def show(self, frac: float, text: str) -> None:
        """Raises BrokenPipeError if the window has gone."""
        pct = max(0, min(99, int(frac * 100)))  # 100 auto-closes; sent only by close()
        self.proc.stdin.write(f"# {text}\n{pct}\n")
        self.proc.stdin.flush()

    def close(self, wait: float = 0.0) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.write("100\n")
                self.proc.stdin.close()
            except BrokenPipeError:
                pass
            if wait:
                self.proc.wait(timeout=wait)


def _mac_progress_helper() -> str:
    """Path of the compiled mac_progress.swift, built on first use (one binary per source version;
    the first build takes ~1 min while swiftc fills its module cache). "" if it can't be built:
    then there is no window. Build errors go to the log folder."""
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mac_progress.swift")
    try:
        with open(src, "rb") as f:
            exe = os.path.expanduser(f"~/.cache/ltx-inpaint-still/bin/ltxag_progress-"
                                     f"{hashlib.sha256(f.read()).hexdigest()[:12]}")
        if os.path.exists(exe):
            return exe
        os.makedirs(os.path.dirname(exe), exist_ok=True)
        tmp = f"{exe}.{os.getpid()}.tmp"  # two jobs may build at once; os.replace is atomic
        r = subprocess.run(["/usr/bin/xcrun", "swiftc", "-O", "-o", tmp, src], capture_output=True,
                           text=True, timeout=600)
        if r.returncode != 0:
            logs = os.path.expanduser("~/.cache/ltx-inpaint-still/logs")
            os.makedirs(logs, exist_ok=True)
            with open(os.path.join(logs, "mac_progress_build.log"), "w") as f:
                f.write(r.stdout + r.stderr)
            return ""
        os.replace(tmp, exe)
        return exe
    except (OSError, subprocess.SubprocessError):
        return ""


def message(kind: str, title: str, text: str, width: int, enabled: bool) -> None:
    """A detached message window; kind is "info" or "error". `width` is unused: the dialog sizes
    itself (kept so callers need not change)."""
    # CVT port (cvt1.10): the system test runs under its own HOME and leaves this marker there,
    # so its simulated failures don't pop up dialogs on the user's screen. A real HOME never has it.
    if os.path.exists(os.path.expanduser("~/.ltxag-no-dialogs")) or not enabled:
        return
    icon = "stop" if kind == "error" else "note"
    subprocess.Popen(["/usr/bin/osascript", "-e", "on run argv", "-e",
                      f'display dialog (item 2 of argv) with title (item 1 of argv) '
                      f'buttons {{"OK"}} default button "OK" with icon {icon}',
                      "-e", "end run", title, text],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


def process_name(pid: int) -> str:
    """Short name of a live process, "" if it is gone. ps's "comm" is the executable's path
    (man ps); pid 0 is the kernel."""
    try:
        if int(pid) <= 0:
            return ""
        r = subprocess.run(["/bin/ps", "-p", str(int(pid)), "-o", "comm="], capture_output=True,
                           text=True, timeout=10)
        return os.path.basename(r.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return ""
