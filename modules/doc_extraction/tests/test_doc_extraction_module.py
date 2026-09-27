from fastapi.testclient import TestClient

from llmp.api.app import create_app
from llmp.config import Settings


def test_module_is_discovered_via_entry_point() -> None:
    app = create_app(Settings(_env_file=None, llm_backend="mock"))
    with TestClient(app) as client:
        assert "doc_extraction" in client.get("/health").json()["modules"]
