from openai_test.client import resolve_api_key


def test_resolve_api_key_prefers_pai(monkeypatch):
    monkeypatch.setenv("PAI_OPENAI_API_KEY", "pai-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    assert resolve_api_key() == "pai-key"


def test_resolve_api_key_falls_back(monkeypatch):
    monkeypatch.delenv("PAI_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    assert resolve_api_key() == "openai-key"
