import random
from decimal import Decimal

import pypdfium2 as pdfium

from doc_extraction.synth.generator import generate_chain
from doc_extraction.synth.pdf import it_number, render_pdf
from doc_extraction.synth.scan import ScanLevel, degrade_pdf

INVOICE = generate_chain(42, 3).invoice


def text_of(pdf: bytes) -> str:
    return str(pdfium.PdfDocument(pdf)[0].get_textpage().get_text_range())


def test_italian_number_format() -> None:
    assert it_number(Decimal("1234.5")) == "1.234,50"


def test_native_pdf_is_deterministic_and_has_a_text_layer() -> None:
    for variant in (0, 1):
        pdf = render_pdf(INVOICE, variant)
        assert pdf == render_pdf(INVOICE, variant)
        text = text_of(pdf)
        assert INVOICE.number in text
        assert f"IT{INVOICE.supplier.vat_number}" in text


def test_layout_variants_differ() -> None:
    assert text_of(render_pdf(INVOICE, 0)) != text_of(render_pdf(INVOICE, 1))


def test_scan_is_deterministic_and_has_no_text_layer() -> None:
    pdf = render_pdf(INVOICE, 0)
    for level in ScanLevel:
        scan = degrade_pdf(pdf, level, random.Random(7))
        assert scan == degrade_pdf(pdf, level, random.Random(7))
        assert text_of(scan).strip() == ""
