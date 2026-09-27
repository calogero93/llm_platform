from functools import cache
from importlib.resources import files

from lxml import etree

# Invoices come from outside: no external entities, no DTD loading, no network (XXE-safe).
SAFE_PARSER = etree.XMLParser(
    resolve_entities=False, no_network=True, load_dtd=False, huge_tree=False
)


class InvalidFatturaPAError(ValueError):
    pass


@cache
def _schema() -> etree.XMLSchema:
    xsd_dir = files("doc_extraction.fatturapa") / "xsd"
    # The vendored xmldsig schema has an internal DTD subset defining entities: this trusted,
    # local file needs them resolved (still no network).
    parser = etree.XMLParser(no_network=True, resolve_entities=True)
    doc = etree.parse(str(xsd_dir / "Schema_VFPR12_v1.2.3.xsd"), parser)
    return etree.XMLSchema(doc)


def validate(xml: bytes) -> None:
    """Raise `InvalidFatturaPAError` if the document does not conform to the FPR12 1.2.3 XSD."""
    schema = _schema()
    if not schema.validate(etree.fromstring(xml, SAFE_PARSER)):
        raise InvalidFatturaPAError(str(schema.error_log)[:2000])
