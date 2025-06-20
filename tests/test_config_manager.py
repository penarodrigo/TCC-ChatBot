#test_config_manager.py
# c:\Users\c057545\Downloads\CBCODIMPROVED\tests\test_config_manager.py
import os
import pytest
from rag_gemini_improved import ConfigManager # Importe a classe que você quer testar

def test_config_manager_defaults(monkeypatch):
    """Testa se o ConfigManager carrega valores padrão quando as variáveis de ambiente não estão definidas."""
    # Usa monkeypatch para simular a ausência de variáveis de ambiente
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("CHUNK_SIZE", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)

    config = ConfigManager()

    assert config.google_api_key == "" # Valor padrão definido na classe
    assert config.chunk_size == 300    # Valor padrão definido na classe
    assert config.embedding_model_name == "intfloat/multilingual-e5-large" # Valor padrão

def test_config_manager_with_env_vars(monkeypatch):
    """Testa se o ConfigManager carrega valores das variáveis de ambiente."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test_api_key_from_env")
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("MAX_CONTEXT_DOCS", "10")

    config = ConfigManager()

    assert config.google_api_key == "test_api_key_from_env"
    assert config.chunk_size == 500
    assert config.max_context_docs == 10

def test_config_manager_invalid_chunk_size(monkeypatch):
    """Testa se o ConfigManager levanta um erro para CHUNK_SIZE inválido."""
    monkeypatch.setenv("CHUNK_SIZE", "0")
    with pytest.raises(ValueError, match="CHUNK_SIZE deve ser maior que 0"):
        ConfigManager()
