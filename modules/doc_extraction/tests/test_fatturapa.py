from decimal import Decimal

import pytest
from lxml import etree

from doc_extraction.fatturapa.parser import parse_fatturapa
from doc_extraction.fatturapa.writer import to_fatturapa_xml
from doc_extraction.fatturapa.xml import InvalidFatturaPAError, validate
from doc_extraction.synth.generator import Chain, generate_chain


def xml_for(chain: Chain) -> bytes:
    return to_fatturapa_xml(chain.invoice, "00001", {chain.ddt.number: chain.ddt.date})


@pytest.mark.parametrize("index", range(20))
def test_generated_invoices_are_xsd_valid_and_round_trip(index: int) -> None:
    chain = generate_chain(42, index)
    xml = xml_for(chain)
    validate(xml)
    assert parse_fatturapa(xml) == [chain.invoice]


def test_validator_rejects_schema_violations() -> None:
    xml = xml_for(generate_chain(42, 0))
    postal_code = generate_chain(42, 0).invoice.supplier.postal_code
    broken = xml.replace(f"<CAP>{postal_code}</CAP>".encode(), b"<CAP>ABC</CAP>", 1)
    with pytest.raises(InvalidFatturaPAError, match="CAP"):
        validate(broken)


def test_parser_returns_one_document_per_body() -> None:
    root = etree.fromstring(xml_for(generate_chain(42, 0)))
    body = root.find("FatturaElettronicaBody")
    assert body is not None
    second = etree.fromstring(etree.tostring(body))
    second.find("DatiGenerali/DatiGeneraliDocumento/Numero").text = "SECOND-1"  # type: ignore[union-attr]
    root.append(second)
    docs = parse_fatturapa(etree.tostring(root))
    assert [d.number for d in docs] == [docs[0].number, "SECOND-1"]


def test_missing_quantity_means_one_unit() -> None:
    xml = xml_for(generate_chain(42, 0))
    root = etree.fromstring(xml)
    line = root.find("FatturaElettronicaBody/DatiBeniServizi/DettaglioLinee")
    assert line is not None
    line.remove(line.find("Quantita"))  # type: ignore[arg-type]
    (doc,) = parse_fatturapa(etree.tostring(root))
    assert doc.lines[0].quantity == Decimal(1)


def test_documents_with_a_dtd_are_rejected() -> None:
    xml = xml_for(generate_chain(42, 0)).decode()
    evil = xml.replace(
        "<?xml version='1.0' encoding='UTF-8'?>",
        '<?xml version="1.0"?><!DOCTYPE p [<!ENTITY x SYSTEM "file:///etc/hostname">]>',
    ).replace("<Numero>", "<Numero>&x;", 1)
    with pytest.raises(InvalidFatturaPAError, match="DTD"):
        parse_fatturapa(evil.encode())
