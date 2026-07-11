import importlib
import os


def test_vultr_llm_keeps_provider_prefix(monkeypatch):
    monkeypatch.setenv("VULTR_SERVERLESS_INFERENCE_API_KEY", "dummy-key")
    monkeypatch.delenv("VULTR_INFERENCE_API_KEY", raising=False)
    monkeypatch.delenv("VULTR_OPENAI_PREFIX", raising=False)

    import main

    importlib.reload(main)
    llm = main._vultr_llm("llama-3.3-70b-instruct-fp8", temperature=0.2)

    assert llm.model.startswith("openai/")
    assert llm.litellm_kwargs["model"].startswith("openai/")
