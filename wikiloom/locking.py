"""File-level locking for WikiLoom operations."""

from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path

LOCK_FILE = ".wikiloom.lock"
STALE_THRESHOLD_SECONDS = 300  # 5 minutes


def _is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    try:
        os.kill(pid, 0)  # signal 0 = don't kill, just check existence
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # process exists but we can't signal it


class FileLock:
    """File-level write lock using .wikiloom.lock.

    Supports context manager protocol for safe acquire/release.
    Only clears a lock when the owning process is dead.
    """

    def __init__(self, project_root: Path):
        self.lock_path = project_root / LOCK_FILE
        self._acquired = False

    def acquire(self, timeout_seconds: int = 30) -> bool:
        """Acquire the lock, waiting up to timeout_seconds.

        Returns True if acquired, False if timed out.
        """
        deadline = time.monotonic() + timeout_seconds

        while time.monotonic() < deadline:
            if self.lock_path.exists():
                self._clear_if_dead()

            if not self.lock_path.exists():
                try:
                    fd = os.open(
                        str(self.lock_path),
                        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    )
                    lock_data = json.dumps({
                        "pid": os.getpid(),
                        "timestamp": time.time(),
                    })
                    os.write(fd, lock_data.encode())
                    os.close(fd)
                    self._acquired = True
                    return True
                except FileExistsError:
                    pass

            time.sleep(0.1)

        return False

    def release(self) -> None:
        """Release the lock."""
        if self.lock_path.exists():
            try:
                self.lock_path.unlink()
            except FileNotFoundError:
                pass
        self._acquired = False

    def _clear_if_dead(self) -> None:
        """Remove the lock file only if the owning process is no longer running."""
        try:
            data = json.loads(self.lock_path.read_text(encoding="utf-8"))
            pid = data.get("pid")
            if pid is not None and not _is_pid_alive(pid):
                self.lock_path.unlink(missing_ok=True)
        except (json.JSONDecodeError, OSError):
            # Corrupt lock file — safe to remove
            self.lock_path.unlink(missing_ok=True)

    def __enter__(self) -> FileLock:
        if not self.acquire():
            raise TimeoutError("Could not acquire WikiLoom lock")
        return self

    def __exit__(self, *args: object) -> None:
        self.release()
