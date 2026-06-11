#!/usr/bin/env python3
"""
Russian TTS Studio — кнопка запуска для пользователя.

Системный трей-индикатор (AyatanaAppIndicator3) + дублирующее окно tkinter.
Один клик по иконке — открыть Web UI / показать статус; правый клик — меню.

Использует системный python3, потому что AppIndicator3 жить в venv не умеет.
Сам Web UI стартует из venv (`<project>/.venv/bin/python` или
`<project>/.venv-voxcpm/bin/python` — на выбор пользователя).

Запуск из корня проекта:
    python3 launcher.py                  (в фоне, иконка в трее)
    python3 launcher.py --window         (показать окно настроек)
    python3 launcher.py --install         (поставить в ~/.local/share/applications/)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

# Lazy imports — defer subprocess / urllib until they are actually needed.
# They can pull in networking / DNS init code at import time, which has
# occasionally hung in sandboxed environments.

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PORT = int(os.environ.get("PORT", "8129"))
LOG_DIR = PROJECT_ROOT / "output" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "launcher.log"
STATE_FILE = Path.home() / ".config" / "russian-tts-studio" / "state.json"
ICON_DIR = PROJECT_ROOT / "web" / "static"
ICON_FILE = ICON_DIR / "launcher-icon.png"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# State persistence (last venv, mode, port)
# ---------------------------------------------------------------------------


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    return {}


def save_state(state: dict) -> None:
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except OSError as e:
        log(f"save_state: {e}")


# ---------------------------------------------------------------------------
# venv detection
# ---------------------------------------------------------------------------


def find_venvs() -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    for name in (".venv", ".venv-voxcpm"):
        p = PROJECT_ROOT / name / "bin" / "python"
        if p.exists():
            label = "XTTS v2" if name == ".venv" else "VoxCPM2"
            found.append((f"{label}  ({name})", p))
    return found


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        try:
            s.connect(("127.0.0.1", port))
            return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            return False


def wait_for_http(url: str, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as r:
                if 200 <= r.status < 500:
                    return True
        except (urllib.error.URLError, ConnectionResetError, OSError):
            pass
        time.sleep(0.4)
    return False


# ---------------------------------------------------------------------------
# Server controller (thread-safe)
# ---------------------------------------------------------------------------


class ServerController:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.port: int = DEFAULT_PORT
        self.url: str = ""

    def is_running(self) -> bool:
        with self._lock:
            return self.proc is not None and self.proc.poll() is None

    def status_text(self) -> str:
        if self.is_running():
            return f"Сервер запущен на {self.url}"
        return "Сервер остановлен"

    def start(self, python_path: Path, mode: str, port: int) -> tuple[bool, str]:
        with self._lock:
            if self.is_running():
                return False, "Уже запущен"
            if port_in_use(port):
                return False, f"Порт {port} уже занят"

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            cmd = [str(python_path), "-m", "web.run", "--no-reload", "--port", str(port)]
            log(f"start: {' '.join(cmd)}")
            try:
                self.proc = subprocess.Popen(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                # Write pidfile so `launcher.py --stop` (which does NOT
                # construct a ServerController) can find the child.
                pidfile = Path.home() / ".config" / "russian-tts-studio" / "server.pid"
                try:
                    pidfile.parent.mkdir(parents=True, exist_ok=True)
                    pidfile.write_text(str(self.proc.pid), encoding="utf-8")
                except OSError:
                    pass
            except (OSError, FileNotFoundError) as e:
                self.proc = None
                return False, f"Не удалось запустить: {e}"

            self.port = port
            self.url = f"http://127.0.0.1:{port}/"
            threading.Thread(target=self._pump_log, args=(self.proc.stdout,), daemon=True).start()

        # ждём ответа вне лока, чтобы не блокировать UI
        if not wait_for_http(self.url, timeout=60.0):
            self.stop()
            return False, f"Сервер не ответил за 60 с. См. {LOG_FILE}"

        if mode == "Web":
            self._open_browser(self.url)
        elif mode == "Desktop":
            self._open_desktop(python_path, self.url)
        # mode == "HTTP": ничего не открываем
        return True, self.url

    def _pump_log(self, stream) -> None:
        try:
            for line in iter(stream.readline, ""):
                log(f"  server | {line.rstrip()}")
        except Exception as e:  # noqa: BLE001
            log(f"log pump: {e}")

    def _open_browser(self, url: str) -> None:
        log(f"open browser: {url}")
        if shutil.which("xdg-open"):
            subprocess.Popen(["xdg-open", url], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        else:
            webbrowser.open(url)

    def _open_desktop(self, python_path: Path, url: str) -> None:
        log("open desktop (pywebview)")
        try:
            subprocess.Popen(
                [str(python_path), "-m", "web.desktop", "--url", url],
                cwd=str(PROJECT_ROOT),
            )
        except (OSError, FileNotFoundError) as e:
            log(f"desktop open failed: {e}")

    def stop(self) -> tuple[bool, str]:
        with self._lock:
            if not self.is_running():
                return False, "Уже остановлен"
            assert self.proc is not None
            log(f"stop: pid={self.proc.pid}")
            try:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=3)
            except Exception as e:  # noqa: BLE001
                log(f"stop error: {e}")
            self.proc = None
        # Clear pidfile
        try:
            (Path.home() / ".config" / "russian-tts-studio" / "server.pid").unlink()
        except OSError:
            pass
        return True, "Остановлен"


# ---------------------------------------------------------------------------
# Icon (synthesize once at first run)
# ---------------------------------------------------------------------------


def ensure_icon() -> Path:
    """Create a simple launcher icon if missing."""
    if ICON_FILE.exists():
        return ICON_FILE
    ICON_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import struct
        import zlib

        # 64x64 PNG: a microphone-ish blue circle on transparent
        w = h = 64
        img = bytearray()
        for y in range(h):
            img.append(0)  # filter
            for x in range(w):
                dx, dy = x - 32, y - 32
                r2 = dx * dx + dy * dy
                if r2 < 26 * 26:
                    img += b"\x29\x7a\xe8\xff"  # blue
                elif r2 < 28 * 28:
                    img += b"\x9a\xc0\xf5\xff"  # lighter blue
                else:
                    img += b"\x00\x00\x00\x00"

        def chunk(tag: bytes, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data))
                + tag
                + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
            )

        png = b"\x89PNG\r\n\x1a\n"
        png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
        png += chunk(b"IDAT", zlib.compress(bytes(img)))
        png += chunk(b"IEND", b"")
        ICON_FILE.write_bytes(png)
    except Exception as e:  # noqa: BLE001
        log(f"icon synth failed: {e}; using fallback")
        return Path(__file__).resolve()  # GTK will fall back to default
    return ICON_FILE


# ---------------------------------------------------------------------------
# Tray (AyatanaAppIndicator3 — works on Ubuntu/GNOME, KDE, XFCE, MATE)
# ---------------------------------------------------------------------------


def _have_appindicator() -> bool:
    try:
        import gi

        gi.require_version("AyatanaAppIndicator3", "0.1")
        from gi.repository import AyatanaAppIndicator3  # noqa: F401

        gi.require_version("Gtk", "3.0")
        from gi.repository import Gtk  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def _have_tk() -> bool:
    try:
        import tkinter  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def run_tray(server: ServerController, state: dict) -> int:
    import gi

    gi.require_version("Gtk", "3.0")
    gi.require_version("AyatanaAppIndicator3", "0.1")
    from gi.repository import GLib, Gtk

    icon_path = str(ensure_icon())
    ind = gi.repository.AyatanaAppIndicator3.Indicator.new(
        "russian-tts-studio",
        icon_path,
        gi.repository.AyatanaAppIndicator3.IndicatorCategory.APPLICATION_STATUS,
    )
    ind.set_status(gi.repository.AyatanaAppIndicator3.IndicatorStatus.ACTIVE)
    ind.set_title("Russian TTS Studio")

    menu = Gtk.Menu()
    mi_status = Gtk.MenuItem(label=server.status_text())
    mi_status.set_sensitive(False)
    menu.append(mi_status)
    menu.append(Gtk.SeparatorMenuItem())

    mi_open = Gtk.MenuItem(label="Открыть Web UI в браузере")
    mi_open.connect("activate", lambda *_: (open_browser() if server.is_running() else start()))
    menu.append(mi_open)
    menu.append(Gtk.SeparatorMenuItem())

    def add_item(label: str, cb) -> None:
        mi = Gtk.MenuItem(label=label)
        mi.connect("activate", lambda *_: cb())
        menu.append(mi)

    def refresh_status() -> None:
        mi_status.set_label(server.status_text())

    def start() -> None:
        venvs = find_venvs()
        if not venvs:
            log("start: no venv")
            return
        # pick the one that matches state, fallback to first
        chosen = next((p for lbl, p in venvs if state.get("venv") and state["venv"] in lbl), venvs[0][1])
        mode = state.get("mode", "Web")
        port = int(state.get("port", DEFAULT_PORT))
        ok, msg = server.start(chosen, mode, port)
        log(f"start -> {ok}: {msg}")
        GLib.idle_add(refresh_status)

    def stop() -> None:
        server.stop()
        GLib.idle_add(refresh_status)

    def open_browser() -> None:
        if server.is_running():
            webbrowser.open(server.url)

    def open_settings() -> None:
        # Run settings window in a new process so tray stays responsive
        subprocess.Popen([sys.executable, str(PROJECT_ROOT / "launcher.py"), "--window"])

    def quit_app(_) -> None:
        log("quit")
        server.stop()
        Gtk.main_quit()

    add_item("Запустить Web UI", start)
    mi_stop = Gtk.MenuItem(label="Остановить сервер")
    mi_stop.connect("activate", lambda *_: stop())
    menu.append(mi_stop)
    menu.append(Gtk.SeparatorMenuItem())
    add_item("Настройки…", open_settings)
    add_item("Выход", lambda: quit_app(None))

    menu.show_all()

    # NOTE: AyatanaAppIndicator3 has no "activate" signal (only on legacy
    # AppIndicator3). Left-click on Ayatana always opens the menu. Add a
    # "Открыть в браузере" item as the first menu entry instead.
    ind.set_secondary_activate_target(mi_open)  # middle-click → open browser

    # Periodic status refresh (1 Hz) so the menu label updates
    def tick() -> bool:
        refresh_status()
        return True

    GLib.timeout_add(1000, tick)

    log("tray ready")
    Gtk.main()
    return 0


# ---------------------------------------------------------------------------
# Settings window (tkinter, fallback GUI)
# ---------------------------------------------------------------------------


def run_window(server: ServerController, state: dict) -> int:
    if not _have_tk():
        print("❌ tkinter недоступен и AppIndicator не работает.", file=sys.stderr)
        print("   Запустите Web UI вручную: .venv/bin/python -m web.run", file=sys.stderr)
        return 1
    # Avoid a 2-min hang when run from a headless terminal — fail fast instead.
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        print("❌ Нет $DISPLAY/WAYLAND_DISPLAY — окно негде показать.", file=sys.stderr)
        print("   Запустите из графической сессии или:", file=sys.stderr)
        print("     .venv/bin/python -m web.run --port 8129", file=sys.stderr)
        return 1
    import tkinter as tk
    from tkinter import messagebox, ttk

    venvs = find_venvs()
    if not venvs:
        print("❌ В проекте нет ни .venv, ни .venv-voxcpm.", file=sys.stderr)
        return 1

    root = tk.Tk()
    root.title("Russian TTS Studio")
    root.geometry("480x280")
    root.minsize(440, 260)

    pad = {"padx": 12, "pady": 6}
    ttk.Label(root, text="Russian TTS Studio", font=("TkDefaultFont", 14, "bold")).pack(anchor="w", **pad)

    f1 = ttk.Frame(root); f1.pack(fill="x", **pad)
    ttk.Label(f1, text="Окружение:").pack(side="left")
    venv_var = tk.StringVar(value=state.get("venv", venvs[0][0]))
    ttk.Combobox(f1, textvariable=venv_var, values=[v[0] for v in venvs], state="readonly", width=28).pack(side="left", padx=8)

    f2 = ttk.Frame(root); f2.pack(fill="x", **pad)
    ttk.Label(f2, text="Режим:").pack(side="left")
    mode_var = tk.StringVar(value=state.get("mode", "Web"))
    ttk.Combobox(f2, textvariable=mode_var,
                 values=["Web", "Desktop", "HTTP"], state="readonly", width=10).pack(side="left", padx=8)

    f3 = ttk.Frame(root); f3.pack(fill="x", **pad)
    ttk.Label(f3, text="Порт:").pack(side="left")
    port_var = tk.StringVar(value=str(state.get("port", DEFAULT_PORT)))
    ttk.Entry(f3, textvariable=port_var, width=8).pack(side="left", padx=8)

    status_var = tk.StringVar(value=server.status_text())
    ttk.Label(root, textvariable=status_var, foreground="#444").pack(anchor="w", padx=12, pady=(0, 4))

    btns = ttk.Frame(root); btns.pack(fill="x", padx=12, pady=14)
    start_btn = ttk.Button(btns, text="▶  Запустить", width=14)
    start_btn.pack(side="left", padx=(0, 6))
    stop_btn = ttk.Button(btns, text="■  Остановить", width=14)
    stop_btn.pack(side="left", padx=6)
    ttk.Button(btns, text="Открыть в браузере", command=lambda: webbrowser.open(server.url) if server.is_running() else None).pack(side="left", padx=6)

    def refresh() -> None:
        status_var.set(server.status_text())
        root.after(500, refresh)

    def do_start() -> None:
        try:
            port = int(port_var.get())
        except ValueError:
            messagebox.showerror("Порт", "Порт должен быть числом", parent=root); return
        venv_path = next(p for lbl, p in venvs if lbl == venv_var.get())
        save_state({"venv": venv_var.get(), "mode": mode_var.get(), "port": port})
        ok, msg = server.start(venv_path, mode_var.get(), port)
        if not ok:
            messagebox.showerror("Не запустилось", msg, parent=root)

    def do_stop() -> None:
        server.stop()

    start_btn.configure(command=do_start)
    stop_btn.configure(command=do_stop)
    root.protocol("WM_DELETE_WINDOW", lambda: (server.stop(), root.destroy()))

    refresh()
    root.mainloop()
    return 0


# ---------------------------------------------------------------------------
# .desktop installation
# ---------------------------------------------------------------------------


def install_desktop() -> int:
    apps = Path.home() / ".local" / "share" / "applications"
    apps.mkdir(parents=True, exist_ok=True)
    target = apps / "russian-tts-studio.desktop"
    icon_path = ensure_icon()
    desktop = f"""[Desktop Entry]
Type=Application
Name=Russian TTS Studio
Name[ru]=Russian TTS Studio
GenericName=TTS Web UI Launcher
Comment=Запустить Web UI для синтеза русской речи
Exec={shutil.which('python3') or 'python3'} "{PROJECT_ROOT / 'launcher.py'}"
Icon={icon_path}
Terminal=false
Categories=Audio;Utility;Development;
StartupNotify=true
Keywords=tts;voice;speech;russian;
"""
    target.write_text(desktop, encoding="utf-8")
    target.chmod(0o755)
    log(f"installed: {target}")
    if shutil.which("update-desktop-database"):
        import subprocess
        subprocess.run(["update-desktop-database", str(apps)], check=False)
    print(f"✓ Установлено: {target}")
    print(f"  Иконка:        {icon_path}")
    print(f"  Запускается:   {shutil.which('python3') or 'python3'} launcher.py")
    print(f"\nИщите «Russian TTS Studio» в меню приложений или на рабочем столе.")
    return 0


def _stop_running_server() -> int:
    """Kill any running uvicorn for this project — minimal, no GUI, no state.

    Two strategies:
      1) Read the pidfile (written by ServerController.start).
      2) Fallback: pgrep for `python.*web.run --port <default>` and SIGTERM.
    """
    import signal
    import subprocess as _sp

    pidfile = Path.home() / ".config" / "russian-tts-studio" / "server.pid"
    killed = 0

    def _kill_pid(pid: int) -> bool:
        try:
            os.kill(pid, signal.SIGTERM)
            return True
        except (OSError, ProcessLookupError):
            return False

    if pidfile.exists():
        try:
            pid = int(pidfile.read_text(encoding="utf-8").strip())
            if _kill_pid(pid):
                killed += 1
                print(f"✓ SIGTERM → pid {pid} (из pidfile)")
        except (OSError, ValueError):
            pass
        try:
            pidfile.unlink()
        except OSError:
            pass

    # Fallback: scan process list
    try:
        out = _sp.run(
            ["pgrep", "-f", "web\\.run --(no-)?reload --port"],
            capture_output=True, text=True, timeout=2,
        )
        for line in out.stdout.splitlines():
            line = line.strip()
            if line.isdigit() and int(line) != os.getpid():
                if _kill_pid(int(line)):
                    killed += 1
                    print(f"✓ SIGTERM → pid {line} (pgrep)")
    except (OSError, _sp.TimeoutExpired, FileNotFoundError):
        pass

    if killed == 0:
        print("– сервер не запущен (или уже остановлен)")
        return 1
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Russian TTS Studio launcher")
    ap.add_argument("--window", action="store_true", help="Показать окно настроек (tkinter)")
    ap.add_argument("--install", action="store_true", help="Установить .desktop-файл в меню приложений")
    ap.add_argument("--stop", action="store_true", help="Остановить запущенный сервер и выйти")
    args = ap.parse_args()

    if args.install:
        return install_desktop()

    # --stop: do the absolute minimum — no state load, no controller
    # construction, just kill any running uvicorn.
    if args.stop:
        return _stop_running_server()

    state = load_state()
    server = ServerController()

    if args.window or not _have_appindicator():
        if args.window:
            return run_window(server, state)
        print("AppIndicator недоступен — открываю окно tkinter.")
        print("   Подсказка: для иконки в трее установите gir1.2-ayatanaappindicator3-0.1")
        return run_window(server, state)

    return run_tray(server, state)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
