import hashlib
import io
import json
from zipfile import ZipFile

import pytest

import scripts.download_data as download_data


class _FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes, headers: dict[str, str]):
        super().__init__(payload)
        self.headers = headers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def _dataset_archive() -> tuple[bytes, dict[str, bytes]]:
    files = {}
    for filename in download_data.REQUIRED_FILES:
        columns = download_data.REQUIRED_COLUMNS.get(filename)
        if columns:
            content = (",".join(sorted(columns)) + "\n").encode()
        else:
            content = f"new content for {filename}\n".encode()
        files[filename] = content

    archive_buffer = io.BytesIO()
    with ZipFile(archive_buffer, "w") as archive:
        for filename, content in files.items():
            archive.writestr(f"dataset/{filename}", content)
    return archive_buffer.getvalue(), files


def _snapshot(directory):
    return {
        path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()
    }


def test_download_failure_leaves_existing_raw_data_unchanged(tmp_path, monkeypatch):
    output_dir = tmp_path / "raw"
    output_dir.mkdir()
    for filename in download_data.REQUIRED_FILES:
        (output_dir / filename).write_bytes(f"old content for {filename}\n".encode())
    (output_dir / "data_manifest.json").write_text(
        '{"snapshot": "old"}\n', encoding="utf-8"
    )
    before = _snapshot(output_dir)
    archive_payload, _ = _dataset_archive()

    monkeypatch.setattr(
        download_data,
        "urlopen",
        lambda *_args, **_kwargs: _FakeResponse(
            archive_payload,
            {
                "ETag": '"new-etag"',
                "Last-Modified": "Sat, 09 Aug 2026 12:00:00 GMT",
            },
        ),
    )

    def fail_validation(_staging_dir):
        raise RuntimeError("staged schema validation failed")

    monkeypatch.setattr(download_data, "validate_raw_schema", fail_validation)

    with pytest.raises(RuntimeError, match="staged schema validation failed"):
        download_data.download_data(output_dir)

    assert _snapshot(output_dir) == before


def test_download_success_replaces_all_raw_files_and_preserves_provenance(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "raw"
    output_dir.mkdir()
    for filename in download_data.REQUIRED_FILES:
        (output_dir / filename).write_bytes(f"old content for {filename}\n".encode())
    (output_dir / "data_manifest.json").write_text(
        '{"snapshot": "old"}\n', encoding="utf-8"
    )
    (output_dir / "unrelated.txt").write_text("keep me\n", encoding="utf-8")
    archive_payload, expected_files = _dataset_archive()
    headers = {
        "ETag": '"new-etag"',
        "Last-Modified": "Sat, 09 Aug 2026 12:00:00 GMT",
    }

    monkeypatch.setattr(
        download_data,
        "urlopen",
        lambda *_args, **_kwargs: _FakeResponse(archive_payload, headers),
    )

    download_data.download_data(output_dir)

    for filename, expected_content in expected_files.items():
        assert (output_dir / filename).read_bytes() == expected_content
    assert (output_dir / "unrelated.txt").read_text(encoding="utf-8") == "keep me\n"

    manifest = json.loads((output_dir / "data_manifest.json").read_text())
    assert manifest["source_etag"] == headers["ETag"]
    assert manifest["source_last_modified"] == headers["Last-Modified"]
    assert manifest["archive_sha256"] == hashlib.sha256(archive_payload).hexdigest()
    assert set(manifest["files"]) == set(download_data.REQUIRED_FILES)
