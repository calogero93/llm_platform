import json
from pathlib import Path

import pytest

import doc_extraction.synth.dataset as dataset
from doc_extraction.synth.dataset import build_split, check_split, load_record


@pytest.fixture
def tiny(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(dataset, "SPLITS", {"ci": 2})
    build_split(tmp_path, "ci", seed=42)
    return tmp_path


def test_split_has_documents_truth_and_manifest(tiny: Path) -> None:
    chain_dir = tiny / "ci" / "chain-0001"
    names = sorted(p.name for p in chain_dir.iterdir())
    assert names == sorted(
        ["order.pdf", "order.scan.pdf", "ddt.pdf", "ddt.scan.pdf", "invoice.pdf",
         "invoice.scan.pdf", "invoice.xml", "truth.json"]
    )  # fmt: skip
    record = load_record(chain_dir)
    assert record.chain.chain_id == "chain-0001"
    manifest = json.loads((tiny / "ci" / "manifest.json").read_text())
    assert len(manifest["files"]) == 16


def test_regeneration_matches_manifest(tiny: Path) -> None:
    assert check_split(tiny, "ci") == []


def test_check_reports_files_that_changed(tiny: Path) -> None:
    path = tiny / "ci" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"]["chain-0000/invoice.xml"] = "0" * 64
    path.write_text(json.dumps(manifest))
    assert check_split(tiny, "ci") == ["chain-0000/invoice.xml"]
