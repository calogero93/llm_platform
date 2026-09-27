"""Build a versioned dataset split on disk: documents + ground truth + manifest of hashes.

Layout: `<root>/<split>/chain-NNNN/{order,ddt,invoice}.pdf, *.scan.pdf, invoice.xml, truth.json`
and `<root>/<split>/manifest.json`. Only `truth.json` and manifests are committed; documents are
regenerated deterministically and verified against the manifest hashes.
"""

import hashlib
import json
import random
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from doc_extraction.fatturapa.writer import to_fatturapa_xml
from doc_extraction.fatturapa.xml import validate
from doc_extraction.synth.generator import Chain, generate_chain
from doc_extraction.synth.pdf import render_pdf
from doc_extraction.synth.scan import ScanLevel, degrade_pdf

DATASET_NAME = "chains"
DATASET_VERSION = 1
SPLITS = {"ci": 30, "full": 300}
SCAN_CYCLE = (ScanLevel.CLEAN, ScanLevel.LIGHT, ScanLevel.HEAVY)


class ChainRecord(BaseModel):
    """Ground truth of one chain plus how its documents were rendered."""

    model_config = ConfigDict(frozen=True)

    chain: Chain
    layout_variant: int
    scan_level: ScanLevel


class Manifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset: str
    version: int
    split: str
    seed: int
    chains: int
    files: dict[str, str]
    """Relative path → SHA-256."""


def build_split(root: Path, split: str, seed: int) -> Manifest:
    split_dir = root / split
    files: dict[str, str] = {}
    for index in range(SPLITS[split]):
        for rel, content in _chain_files(seed, index).items():
            path = split_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            files[rel] = hashlib.sha256(content).hexdigest()
    manifest = Manifest(
        dataset=DATASET_NAME,
        version=DATASET_VERSION,
        split=split,
        seed=seed,
        chains=SPLITS[split],
        files=files,
    )
    (split_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
    return manifest


def load_record(chain_dir: Path) -> ChainRecord:
    return ChainRecord.model_validate_json((chain_dir / "truth.json").read_text())


def _chain_files(seed: int, index: int) -> dict[str, bytes]:
    chain = generate_chain(seed, index)
    variant = index % 2
    level = SCAN_CYCLE[index % len(SCAN_CYCLE)]
    prefix = chain.chain_id
    out: dict[str, bytes] = {}

    xml = to_fatturapa_xml(chain.invoice, f"{index:05d}", {chain.ddt.number: chain.ddt.date})
    validate(xml)  # never ship an invalid invoice as ground truth
    out[f"{prefix}/invoice.xml"] = xml

    documents = {"order": (chain.order, 1 - variant), "ddt": (chain.ddt, variant),
                 "invoice": (chain.invoice, variant)}  # fmt: skip
    for name, (doc, doc_variant) in documents.items():
        pdf = render_pdf(doc, doc_variant)
        out[f"{prefix}/{name}.pdf"] = pdf
        scan_rng = random.Random(f"{seed}:{index}:scan:{name}")
        out[f"{prefix}/{name}.scan.pdf"] = degrade_pdf(pdf, level, scan_rng)

    record = ChainRecord(chain=chain, layout_variant=variant, scan_level=level)
    out[f"{prefix}/truth.json"] = (record.model_dump_json(indent=2) + "\n").encode()
    return out


def check_split(root: Path, split: str) -> list[str]:
    """Regenerate in memory and return the files whose hash differs from the manifest."""
    manifest = Manifest.model_validate(json.loads((root / split / "manifest.json").read_text()))
    actual: dict[str, str] = {}
    for index in range(manifest.chains):
        for rel, content in _chain_files(manifest.seed, index).items():
            actual[rel] = hashlib.sha256(content).hexdigest()
    paths = set(actual) | set(manifest.files)
    return sorted(p for p in paths if actual.get(p) != manifest.files.get(p))
