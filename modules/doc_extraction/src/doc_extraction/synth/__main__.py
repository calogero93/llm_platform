"""`python -m doc_extraction.synth {build,check,contact-sheet} --root <dataset version dir>`."""

import argparse
import sys
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageDraw

from doc_extraction.synth.dataset import SPLITS, build_split, check_split

SEED = 42


def contact_sheet(root: Path, out: Path) -> None:
    """2x2 sheet of the top of invoices: both layouts, then a light and a heavy scan."""
    samples = [
        ("chain-0000/invoice.pdf", "layout 0, native"),
        ("chain-0001/invoice.pdf", "layout 1, native"),
        ("chain-0001/invoice.scan.pdf", "scan: light"),
        ("chain-0002/invoice.scan.pdf", "scan: heavy"),
    ]
    tiles = []
    for rel, label in samples:
        page = pdfium.PdfDocument((root / "ci" / rel).read_bytes())[0]
        image = page.render(scale=110 / 72).to_pil().convert("L")
        tile = image.crop((0, 0, image.width, int(image.height * 0.5)))
        ImageDraw.Draw(tile).text((10, 10), label, fill=0)
        tiles.append(tile)
    w, h = tiles[0].size
    sheet = Image.new("L", (w * 2, h * 2), 255)
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % 2) * w, (i // 2) * h))
    sheet.save(out, quality=80, optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser(prog="doc_extraction.synth")
    parser.add_argument("command", choices=["build", "check", "contact-sheet"])
    parser.add_argument("--root", type=Path, required=True, help="dataset version directory")
    parser.add_argument("--split", choices=sorted(SPLITS), default="ci")
    args = parser.parse_args()

    if args.command == "build":
        manifest = build_split(args.root, args.split, SEED)
        print(f"built {args.split}: {manifest.chains} chains, {len(manifest.files)} files")
    elif args.command == "check":
        diff = check_split(args.root, args.split)
        if diff:
            print(f"{len(diff)} files differ from manifest, e.g. {diff[:5]}", file=sys.stderr)
            return 1
        print(f"{args.split}: regenerated files match manifest")
    else:
        contact_sheet(args.root, args.root / "contact_sheet.jpg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
