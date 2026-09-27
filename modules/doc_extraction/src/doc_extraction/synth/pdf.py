"""Render documents as native (text-layer) PDFs, in two layout variants.

Variants differ in labels, date/number formats and column order, so extraction is not measured
on a single template.
"""

import io
from datetime import date
from decimal import Decimal

from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from doc_extraction.schemas import Document, DocumentKind, LineItem, Party

rl_config.invariant = 1  # no timestamps / random ids in the PDF: byte-identical output

MONTHS = (
    "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
    "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre",
)  # fmt: skip

TITLES = {
    DocumentKind.ORDER: ("Ordine d'acquisto", "ORDINE FORNITORE"),
    DocumentKind.DDT: ("Documento di trasporto", "D.D.T. - DOCUMENTO DI TRASPORTO"),
    DocumentKind.INVOICE: ("Fattura", "FATTURA DIFFERITA"),
}

BASE = ParagraphStyle("base", fontName="Helvetica", fontSize=8.5, leading=10.5)
BOLD = ParagraphStyle("bold", parent=BASE, fontName="Helvetica-Bold")
TITLE = ParagraphStyle("title", parent=BOLD, fontSize=13, leading=16)


def it_number(value: Decimal, decimals: int = 2) -> str:
    """Italian formatting: 1234.5 → '1.234,50'."""
    return f"{value:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def it_quantity(value: Decimal) -> str:
    return it_number(value, 0) if value == value.to_integral_value() else it_number(value)


def _date(value: date, variant: int) -> str:
    if variant == 0:
        return value.strftime("%d/%m/%Y")
    return f"{value.day} {MONTHS[value.month - 1]} {value.year}"


def _party(party: Party, heading: str) -> Paragraph:
    title = f"<b>{heading}</b><br/>" if heading else ""
    return Paragraph(
        f"{title}<b>{party.name}</b><br/>{party.address}<br/>"
        f"{party.postal_code} {party.city} ({party.province})<br/>P.IVA IT{party.vat_number}",
        BASE,
    )


def _line_cells(line: LineItem, doc: Document, variant: int) -> list[str | Paragraph]:
    description = Paragraph(line.description, BASE)
    qty = it_quantity(line.quantity)
    unit = line.unit or ""
    if doc.kind is DocumentKind.DDT:
        return [line.code or "", description, unit, qty]
    price = it_number(line.unit_price) if line.unit_price is not None else ""
    total = it_number(line.total) if line.total is not None else ""
    vat = f"{line.vat_rate:.0f}" if line.vat_rate is not None else ""
    if variant == 0:
        cells: list[str | Paragraph] = [line.code or "", description, unit, qty, price, total]
    else:
        cells = [description, line.code or "", qty, unit, f"€ {price}", f"€ {total}"]
    return [*cells, vat] if doc.kind is DocumentKind.INVOICE else cells


def _header_row(doc: Document, variant: int) -> list[str]:
    if doc.kind is DocumentKind.DDT:
        return ["Codice", "Descrizione", "U.M.", "Q.tà"]
    if variant == 0:
        row = ["Codice", "Descrizione", "U.M.", "Q.tà", "Prezzo unit.", "Importo"]
    else:
        row = ["Descrizione articolo", "Cod. art.", "Quantità", "UM", "Prezzo", "Totale riga"]
    return [*row, "IVA %"] if doc.kind is DocumentKind.INVOICE else row


def _widths(doc: Document, variant: int) -> list[float]:
    if doc.kind is DocumentKind.DDT:
        return [25 * mm, 110 * mm, 15 * mm, 20 * mm]
    if variant == 0:
        widths = [20 * mm, 72 * mm, 12 * mm, 15 * mm, 22 * mm, 22 * mm]
    else:
        widths = [72 * mm, 20 * mm, 17 * mm, 10 * mm, 22 * mm, 22 * mm]
    return [*widths, 12 * mm] if doc.kind is DocumentKind.INVOICE else widths


def _references(doc: Document, variant: int) -> str:
    refs = []
    if doc.order_refs:
        label = "Rif. vs. ordine" if variant == 0 else "Vostro ordine n."
        refs.append(f"{label} {', '.join(doc.order_refs)}")
    if doc.ddt_refs:
        label = "Rif. DDT" if variant == 0 else "Documenti di trasporto"
        refs.append(f"{label} {', '.join(doc.ddt_refs)}")
    return " — ".join(refs)


def _totals(doc: Document) -> Table | None:
    rows: list[list[str]] = []
    if doc.kind is DocumentKind.INVOICE:
        for s in doc.vat_summary:
            rows.append([f"Imponibile IVA {s.vat_rate:.0f}%", it_number(s.taxable_amount)])
            rows.append([f"IVA {s.vat_rate:.0f}%", it_number(s.vat_amount)])
        if doc.total_amount is not None:
            rows.append(["TOTALE DOCUMENTO €", it_number(doc.total_amount)])
    elif doc.kind is DocumentKind.ORDER and doc.taxable_amount is not None:
        rows.append(["Totale imponibile €", it_number(doc.taxable_amount)])
    if not rows:
        return None
    table = Table(rows, colWidths=[50 * mm, 30 * mm], hAlign="RIGHT")
    table.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
                ("FONT", (0, -1), (-1, -1), "Helvetica-Bold", 10),
                ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                ("LINEABOVE", (0, -1), (-1, -1), 0.8, colors.black),
            ]
        )
    )
    return table


def render_pdf(doc: Document, variant: int) -> bytes:
    buf = io.BytesIO()
    pdf = SimpleDocTemplate(
        buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=15 * mm,
        bottomMargin=15 * mm, title=f"{doc.kind.value} {doc.number}",
    )  # fmt: skip
    title = TITLES[doc.kind][variant]
    number_label = "n." if variant == 0 else "Numero documento:"
    date_label = "del" if variant == 0 else "Data:"
    customer_heading = "Spett.le" if variant == 0 else "Destinatario / Cliente"
    if doc.kind is DocumentKind.ORDER:
        # Orders are issued by the customer to the supplier.
        parties = [_party(doc.customer, "Committente"), _party(doc.supplier, "Fornitore")]
    else:
        parties = [
            _party(doc.supplier, "Mittente" if variant else ""),
            _party(doc.customer, customer_heading),
        ]

    head = Table([parties], colWidths=[90 * mm, 90 * mm])
    head.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    lines = Table(
        [_header_row(doc, variant)] + [_line_cells(x, doc, variant) for x in doc.lines],
        colWidths=_widths(doc, variant),
        repeatRows=1,
    )
    lines.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.5),
                ("FONT", (0, 1), (-1, -1), "Helvetica", 8.5),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story = [
        head,
        Spacer(1, 8 * mm),
        Paragraph(
            f"{title} {number_label} {doc.number} {date_label} {_date(doc.date, variant)}", TITLE
        ),
        Spacer(1, 2 * mm),
        Paragraph(_references(doc, variant), BASE),
        Spacer(1, 5 * mm),
        lines,
        Spacer(1, 6 * mm),
    ]
    totals = _totals(doc)
    if totals is not None:
        story.append(totals)
    pdf.build(story)
    return buf.getvalue()
