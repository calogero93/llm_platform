from pathlib import Path

import pytest

import doc_extraction.synth.dataset as dataset
from doc_extraction.evals import InvoiceXmlSuite
from doc_extraction.synth.dataset import build_split
from llmp.config import Settings
from llmp.eval.runner import run_suite
from llmp.llm.mock import MockLLMClient
from llmp.modules import PlatformContext

CTX = PlatformContext(
    settings=Settings(_env_file=None, llm_backend="mock"),
    llm=MockLLMClient("schema", model="m"),
)


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(dataset, "SPLITS", {"ci": 3})
    build_split(tmp_path, "ci", seed=42)
    return tmp_path


async def test_xml_suite_scores_perfectly(root: Path) -> None:
    report = await run_suite(InvoiceXmlSuite(root), "ci", CTX)
    assert (report.n_cases, report.n_errors) == (3, 0)
    assert report.micro.f1 == 1.0
    assert report.scores["exact_match"] == 1.0
    assert report.dataset.name == "chains"


def test_missing_documents_point_to_make_synth(root: Path) -> None:
    (root / "ci" / "chain-0000" / "invoice.xml").unlink()
    with pytest.raises(FileNotFoundError, match="make synth"):
        InvoiceXmlSuite(root).case_ids("ci")
