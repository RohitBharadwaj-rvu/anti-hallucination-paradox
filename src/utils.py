"""
src/utils.py — Logging setup, JSONL checkpoint I/O, and key builder.
"""
import json
import logging
import os
import threading
from pathlib import Path
from datetime import datetime


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure root logger and return the 'ahp' (anti-hallucination paradox) logger."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("ahp")
    logger.setLevel(numeric_level)
    return logger


def get_logger(name: str = "ahp") -> logging.Logger:
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Checkpoint key builder
# ---------------------------------------------------------------------------

def make_key(model_name: str, sample_id: str, stage: str, level: str) -> str:
    """
    Build a stable checkpoint key.
    key(model_name, sample_id, stage, level) = f"{model_name}|{sample_id}|{stage}|{level}"
    """
    return f"{model_name}|{sample_id}|{stage}|{level}"


# ---------------------------------------------------------------------------
# JSONL checkpoint I/O (thread-safe)
# ---------------------------------------------------------------------------

_jsonl_locks: dict[str, threading.Lock] = {}
_lock_registry_lock = threading.Lock()


def _get_lock(path: str) -> threading.Lock:
    with _lock_registry_lock:
        if path not in _jsonl_locks:
            _jsonl_locks[path] = threading.Lock()
        return _jsonl_locks[path]


def load_checkpoint(path: str) -> set:
    """
    Load a JSONL checkpoint file and return the set of already-completed keys.
    Each line must be a JSON object with a 'key' field.
    """
    completed: set = set()
    p = Path(path)
    if not p.exists():
        return completed
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                k = obj.get("key", "")
                if k:
                    completed.add(k)
            except json.JSONDecodeError:
                pass  # ignore corrupt lines
    return completed


def append_checkpoint(path: str, record: dict) -> None:
    """
    Thread-safely append a JSON record (with 'key' field) to a JSONL checkpoint file.
    Creates parent directories if needed.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lock = _get_lock(str(p))
    with lock:
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()


def load_jsonl(path: str) -> list[dict]:
    """Load all records from a JSONL file."""
    records = []
    p = Path(path)
    if not p.exists():
        return records
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def write_jsonl(path: str, records: list[dict]) -> None:
    """Write a list of dicts to a JSONL file (overwrites)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def approx_tokens(text: str) -> int:
    """Approximate token count: len(text) / 4 (char/4 fallback). Never exact."""
    return max(1, len(text) // 4)


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
