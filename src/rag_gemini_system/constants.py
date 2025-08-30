# -*- coding: utf-8 -*-
"""
Módulo para armazenar constantes do sistema.

Este arquivo centraliza valores fixos e padrões para facilitar a manutenção e a consistência em todo o projeto.
"""

# Nomes de Modelos Padrão
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
DEFAULT_GEMINI_MODEL = "gemini-pro"

# Configurações Padrão do RAG
DEFAULT_CHUNK_SIZE = 300
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_MAX_CONTEXT_DOCS = 5

# Configurações Padrão do Gemini
DEFAULT_GEMINI_MAX_TOKENS = 2048
DEFAULT_GEMINI_TEMPERATURE = 0.7

# Configurações de Segurança Padrão do Gemini
DEFAULT_GEMINI_SAFETY_SETTINGS = {
    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
}

# Configurações de Cache Padrão
DEFAULT_CACHE_DIR = ".cache_embeddings"
DEFAULT_ENABLE_CACHE = "True"
DEFAULT_CACHE_TTL_SECONDS = 86400  # 24 horas

# Configurações de Diretórios e Arquivos Padrão
DEFAULT_DATA_DIR = "dados_rag"
DEFAULT_VECTOR_STORE_DIR = ".vector_store"
DEFAULT_TARGET_DOCUMENT = "IDDC.pdf"

# Configurações Padrão do Servidor Flask
DEFAULT_FLASK_HOST = "0.0.0.0"
DEFAULT_FLASK_PORT = 5000
