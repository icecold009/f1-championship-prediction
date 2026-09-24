"""Atomic local artifact writes shared by report and release entrypoints."""

import os
import tempfile
import time
from pathlib import Path


def atomic_write_bytes(path: str | Path, content: bytes) -> Path:
    """Publish complete file contents with a same-directory atomic replace."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as temporary_file:
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        for attempt in range(5):
            try:
                os.replace(temporary_path, destination)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.01 * (attempt + 1))
    except BaseException:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return destination


def atomic_write_text(path: str | Path, content: str, encoding: str = "utf-8") -> Path:
    """Publish text without exposing a partially written destination file."""
    return atomic_write_bytes(path, content.encode(encoding))
