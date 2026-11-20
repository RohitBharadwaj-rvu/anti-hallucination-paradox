"""Lightweight detached process launcher & log tailer for Colab SSH."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    # Check zombie status on Linux
    stat_path = Path(f"/proc/{pid}/stat")
    if stat_path.exists():
        try:
            txt = stat_path.read_text()
            state = txt.rsplit(")", 1)[1].strip().split()[0]
            if state == "Z":
                return False
        except Exception:
            pass
    return True


def main() -> None:
    mode = sys.argv[1]
    if mode == "start":
        log_path = Path(sys.argv[2])
        pid_file = Path(sys.argv[3])
        cmd = sys.argv[4:]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        logf = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd="/content/ahp",
            stdin=subprocess.DEVNULL,
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
        pid_file.write_text(str(proc.pid), encoding="utf-8")
        print(f"STARTED pid={proc.pid} log={log_path}")
    elif mode == "status":
        log_path = Path(sys.argv[2])
        pid_file = Path(sys.argv[3])
        n_lines = int(sys.argv[4]) if len(sys.argv) > 4 else 40
        pid = int(pid_file.read_text().strip()) if pid_file.exists() else -1
        alive = pid > 0 and is_alive(pid)
        print(f"STATUS={'RUNNING' if alive else 'FINISHED'} pid={pid}")
        if log_path.exists():
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            print("--- TAIL ---")
            for ln in lines[-n_lines:]:
                print(ln)
        else:
            print("(no log file yet)")
    else:
        raise SystemExit(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
