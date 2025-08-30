# -*- coding: utf-8 -*-
"""
Módulo de Gerenciamento de Configuração
"""

import os
import logging
from pathlib import Path

# Carrega variáveis de ambiente do arquivo .env
from dotenv import load_dotenv

from . import constants

load_dotenv()

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Gerencia todas as configurações do sistema, carregando de variáveis de ambiente.
    """
    def __init__(self):
        logger.debug("Inicializando ConfigManager...")

        # Chaves e Nomes de Modelos
        self.google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
        self.embedding_model_name: str = os.getenv("EMBEDDING_MODEL", constants.DEFAULT_EMBEDDING_MODEL)
        self.gemini_model_name: str = os.getenv("GEMINI_MODEL", constants.DEFAULT_GEMINI_MODEL)

        # Configurações do RAG
        try:
            self.chunk_size: int = int(os.getenv("CHUNK_SIZE", constants.DEFAULT_CHUNK_SIZE))
            self.chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", constants.DEFAULT_CHUNK_OVERLAP))
            self.max_context_docs: int = int(os.getenv("MAX_CONTEXT_DOCS", constants.DEFAULT_MAX_CONTEXT_DOCS))
        except (ValueError, TypeError) as e:
            logger.error(f"Erro ao converter valor de chunking do .env para número: {e}.")
            raise

        # Configurações do Gemini
        try:
            self.gemini_max_tokens: int = int(os.getenv("GEMINI_MAX_TOKENS", constants.DEFAULT_GEMINI_MAX_TOKENS))
            self.gemini_temperature: float = float(os.getenv("GEMINI_TEMPERATURE", constants.DEFAULT_GEMINI_TEMPERATURE))
        except (ValueError, TypeError) as e:
            logger.error(f"Erro ao converter valor do Gemini do .env para número: {e}.")
            raise

        # Gemini Safety Settings
        self.gemini_safety_settings: dict = constants.DEFAULT_GEMINI_SAFETY_SETTINGS

        # Configurações de Cache
        self.cache_dir: str = os.getenv("CACHE_DIR", constants.DEFAULT_CACHE_DIR)
        self.enable_cache: bool = os.getenv("ENABLE_EMBEDDING_CACHE", constants.DEFAULT_ENABLE_CACHE).lower() == "true"
        try:
            self.cache_ttl: int = int(os.getenv("CACHE_TTL_SECONDS", constants.DEFAULT_CACHE_TTL_SECONDS))
        except (ValueError, TypeError) as e:
            logger.error(f"Erro ao converter CACHE_TTL_SECONDS do .env para número: {e}.")
            raise

        # Configurações de Diretórios e Arquivos
        self.data_dir: str = os.getenv("DATA_DIR", constants.DEFAULT_DATA_DIR)
        self.vector_store_dir: str = os.getenv("VECTOR_STORE_DIR", constants.DEFAULT_VECTOR_STORE_DIR)
        self.target_document_name: str = os.getenv("TARGET_DOCUMENT", constants.DEFAULT_TARGET_DOCUMENT)

        # Configurações do Servidor Flask
        self.flask_host: str = os.getenv("FLASK_HOST", constants.DEFAULT_FLASK_HOST)
        try:
            self.flask_port: int = int(os.getenv("FLASK_PORT", constants.DEFAULT_FLASK_PORT))
        except (ValueError, TypeError) as e:
            logger.error(f"Erro ao converter FLASK_PORT do .env para número: {e}.")
            raise

        self._validate_config()
        logger.info("ConfigManager inicializado e configurações validadas.")

    def _validate_config(self):
        """Valida as configurações para garantir que são lógicas e consistentes."""
        logger.debug("Validando configurações...")
        if not self.google_api_key or self.google_api_key == "your_google_api_key_here":
            logger.warning("⚠️ GOOGLE_API_KEY não está configurada ou é um placeholder.")

        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE deve ser maior que 0")
        if self.chunk_overlap < 0:
            raise ValueError("CHUNK_OVERLAP não pode ser negativo")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP deve ser menor que CHUNK_SIZE")
        if self.max_context_docs <= 0:
            raise ValueError("MAX_CONTEXT_DOCS deve ser maior que 0")

        # Garante que os diretórios de dados existam
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.vector_store_dir).mkdir(parents=True, exist_ok=True)
        logger.debug("Validação de configurações concluída.")

