from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from src import atomic_artifacts


def test_atomic_write_replaces_file_with_complete_content(tmp_path):
    destination = tmp_path / "artifact.html"
    destination.write_bytes(b"previous artifact")

    result = atomic_artifacts.atomic_write_bytes(destination, b"complete artifact")

    assert result == destination
    assert destination.read_bytes() == b"complete artifact"
    assert list(tmp_path.glob(".artifact.html.*.tmp")) == []


def test_failed_atomic_replace_preserves_previous_file(tmp_path, monkeypatch):
    destination = tmp_path / "manifest.json"
    destination.write_bytes(b"previous manifest")

    def fail_replace(source, target):
        assert source.read_bytes() == b"complete new manifest"
        assert target.read_bytes() == b"previous manifest"
        raise OSError("synthetic publication interruption")

    monkeypatch.setattr(atomic_artifacts.os, "replace", fail_replace)

    with pytest.raises(OSError, match="synthetic publication interruption"):
        atomic_artifacts.atomic_write_bytes(destination, b"complete new manifest")

    assert destination.read_bytes() == b"previous manifest"
    assert list(tmp_path.glob(".manifest.json.*.tmp")) == []


def test_atomic_replace_retries_transient_windows_permission_error(
    tmp_path, monkeypatch
):
    destination = tmp_path / "retry.bin"
    real_replace = atomic_artifacts.os.replace
    attempts = 0

    def fail_twice(source, target):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("synthetic transient replace contention")
        real_replace(source, target)

    monkeypatch.setattr(atomic_artifacts.os, "replace", fail_twice)

    atomic_artifacts.atomic_write_bytes(destination, b"published complete")

    assert attempts == 3
    assert destination.read_bytes() == b"published complete"


def test_concurrent_atomic_writers_publish_only_complete_payloads(tmp_path):
    destination = tmp_path / "shared.bin"
    payloads = [bytes([value]) * 500_000 for value in range(1, 7)]

    with ThreadPoolExecutor(max_workers=len(payloads)) as executor:
        list(
            executor.map(
                lambda payload: atomic_artifacts.atomic_write_bytes(
                    destination, payload
                ),
                payloads,
            )
        )

    assert destination.read_bytes() in payloads
    assert list(tmp_path.glob(".shared.bin.*.tmp")) == []


def test_destination_stays_complete_while_writer_is_interrupted(tmp_path, monkeypatch):
    destination = tmp_path / "report.html"
    previous = b"previous complete report"
    replacement = b"new complete report" * 100
    destination.write_bytes(previous)
    staged = Event()
    publish = Event()
    real_replace = atomic_artifacts.os.replace

    def pause_before_publish(source, target):
        assert source.read_bytes() == replacement
        staged.set()
        if not publish.wait(timeout=5):
            raise TimeoutError("synthetic publish remained interrupted")
        real_replace(source, target)

    monkeypatch.setattr(atomic_artifacts.os, "replace", pause_before_publish)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            atomic_artifacts.atomic_write_bytes, destination, replacement
        )
        assert staged.wait(timeout=5)
        try:
            assert destination.read_bytes() == previous
        finally:
            publish.set()
        future.result(timeout=5)

    assert destination.read_bytes() == replacement
