"""Silent background Colab Jupyter-kernel keepalive loop (runs locally via WSL)."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STOP_FLAG = ROOT / "outputs/STOP_KEEPALIVE"
LOG_FILE = ROOT / "outputs/keepalive.log"
KEEPALIVE_PY = "/mnt/c/Users/rohit/anti-hallucination-paradox/stage2_lace_grpo/keepalive.py"


def main() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    if STOP_FLAG.exists():
        STOP_FLAG.unlink()

    consec_fail = 0
    while True:
        if STOP_FLAG.exists():
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Stop flag detected; exiting.\n")
            break
        try:
            res = subprocess.run(
                [
                    "wsl",
                    "bash",
                    "-lc",
                    f"/home/rohit/.local/bin/colab sessions >/dev/null 2>&1 || true; timeout 30 /home/rohit/.local/bin/colab exec -s lace-gpu -f {KEEPALIVE_PY} </dev/null",
                ],
                capture_output=True,
                text=True,
                timeout=45,
            )
            msg = (res.stdout or res.stderr or "").strip().splitlines()[-1:] or [""]
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] rc={res.returncode} {msg[0]}\n")
            if res.returncode == 0:
                consec_fail = 0
            else:
                consec_fail += 1
        except Exception as e:
            consec_fail += 1
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] err={e}\n")

        if consec_fail >= 5:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 5 consecutive failures; exiting.\n")
            break

        for _ in range(48):  # 48 * 10s = 480s (8 min)
            if STOP_FLAG.exists():
                break
            time.sleep(10)


if __name__ == "__main__":
    main()
