"""Eval suites of the document extraction module."""

import hashlib
from pathlib import Path

from doc_extraction.fatturapa.parser import parse_fatturapa
from doc_extraction.metrics import compare_documents
from doc_extraction.synth.dataset import DATASET_NAME, DATASET_VERSION, load_record
from llmp.eval import CaseResult, DatasetRef
from llmp.modules import PlatformContext

# Datasets live in the source tree (modules/doc_extraction/evals/...), not in the wheel.
DATASET_ROOT = (
    Path(__file__).resolve().parents[2]
    / "evals"
    / "datasets"
    / DATASET_NAME
    / f"v{DATASET_VERSION}"
)


class InvoiceXmlSuite:
    """FatturaPA XML → Document with the deterministic parser. Expected F1 = 1.0: anything less
    is a bug in the parser, the generator or the metrics."""

    name = "doc_extraction.invoice_xml"
    gated_metrics = ("micro.f1", "scores.exact_match")

    def __init__(self, root: Path = DATASET_ROOT) -> None:
        self._root = root

    def dataset(self, split: str) -> DatasetRef:
        manifest = (self._root / split / "manifest.json").read_bytes()
        return DatasetRef(
            name=DATASET_NAME,
            version=DATASET_VERSION,
            split=split,
            manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        )

    def case_ids(self, split: str) -> list[str]:
        cases = sorted(p.name for p in (self._root / split).glob("chain-*") if p.is_dir())
        missing = [c for c in cases if not (self._root / split / c / "invoice.xml").exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} cases lack documents (e.g. {missing[0]}): run `make synth`"
            )
        return cases

    async def run_case(self, split: str, case_id: str, ctx: PlatformContext) -> CaseResult:
        chain_dir = self._root / split / case_id
        gold = load_record(chain_dir).chain.invoice
        (pred,) = parse_fatturapa((chain_dir / "invoice.xml").read_bytes())
        counts = compare_documents(gold, pred)
        exact = all(c.fp == 0 and c.fn == 0 for c in counts.values())
        return CaseResult(case_id=case_id, counts=counts, scores={"exact_match": float(exact)})
