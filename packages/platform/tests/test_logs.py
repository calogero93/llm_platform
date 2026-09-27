import json
import logging

from llmp.logs import JsonFormatter, correlation_id


def format_record(**extra: object) -> dict[str, object]:
    record = logging.makeLogRecord(
        {"name": "x", "levelname": "INFO", "msg": "hi %s", "args": ("a",)}
    )
    record.__dict__.update(extra)
    return json.loads(JsonFormatter().format(record))  # type: ignore[no-any-return]


def test_standard_fields_and_extras() -> None:
    entry = format_record(doc_id="d1")
    assert entry["message"] == "hi a"
    assert entry["level"] == "INFO"
    assert entry["doc_id"] == "d1"
    assert entry["correlation_id"] is None
    assert "args" not in entry


def test_includes_current_correlation_id() -> None:
    token = correlation_id.set("cid-1")
    try:
        assert format_record()["correlation_id"] == "cid-1"
    finally:
        correlation_id.reset(token)
