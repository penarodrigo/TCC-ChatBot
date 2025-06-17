# -*- coding: utf-8 -*-
"""
Sistema RAG com Google Gemini para VSCode
Versão Revisada e Melhorada
Adaptado do Workshop Energy GPT - RAG
"""

import os
import logging
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime
import traceback

# Importações para processamento de documentos
import pandas as pd
import pdfplumber
from pptx import Presentation
import docx
from openpyxl import load_workbook

# Importações para IA e embeddings
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Importações para API web
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import json

# Configuração de ambiente
from dotenv import load_dotenv
load_dotenv()

# Configuração de logging melhorada
def setup_logging():
    """Configura o sistema de logging"""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_file = os.getenv("LOG_FILE", "")
    
    # Formato do log
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configuração básica
    handlers = [logging.StreamHandler()]
    
    # Adicionar arquivo se especificado
    if log_file and log_file.strip():
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=handlers
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()

class ConfigManager:
    """Gerenciador de configurações centralizadas"""
    
    def __init__(self):
        self.google_api_key = os.getenv("GOOGLE_API_KEY", "")
        self.chunk_size = int(os.getenv("CHUNK_SIZE", 300))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 50))
        self.max_context_docs = int(os.getenv("MAX_CONTEXT_DOCS", 5))
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
        self.gemini_model_name = os.getenv("GEMINI_MODEL", "gemini-pro")
        self.data_dir = os.getenv("DATA_DIR", "dados_rag")
        self.cache_dir = os.getenv("CACHE_DIR", ".cache")
        self.enable_cache = os.getenv("ENABLE_EMBEDDING_CACHE", "True").lower() == "true"
        self.cache_ttl = int(os.getenv("CACHE_TTL", 86400))
        self.gemini_temperature = float(os.getenv("GEMINI_TEMPERATURE", 0.7))
        self.gemini_max_tokens = int(os.getenv("GEMINI_MAX_TOKENS", 1024))
        
        # Validar configurações críticas
        self._validate_config()
    
    def _validate_config(self):
        """Valida configurações críticas"""
        if not self.google_api_key or self.google_api_key == "your_google_api_key_here":
            logger.warning("⚠️  GOOGLE_API_KEY não configurada corretamente")
        
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE deve ser maior que 0")
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP deve ser menor que CHUNK_SIZE")

class CacheManager:
    """Gerenciador de cache para embeddings"""
    
    def __init__(self, cache_dir: str, ttl: int = 86400):
        self.cache_dir = Path(cache_dir)
        self.ttl = ttl
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_path(self, key: str) -> Path:
        """Gera caminho do arquivo de cache"""
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{hash_key}.pkl"
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Recupera embedding do cache"""
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
        
        # Verificar se não expirou
        if (datetime.now().timestamp() - cache_path.stat().st_mtime) > self.ttl:
            cache_path.unlink()
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Erro ao carregar cache: {e}")
            return None
    
    def set(self, key: str, value: np.ndarray):
        """Salva embedding no cache"""
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.error(f"Erro ao salvar cache: {e}")

class DocumentProcessor:
    """Classe melhorada para processar diferentes tipos de documentos"""
    
    SUPPORTED_EXTENSIONS = {'.txt', '.pdf', '.pptx', '.csv', '.docx', '.xlsx'}
    
    @classmethod
    def is_supported(cls, file_path: Union[str, Path]) -> bool:
        """Verifica se o arquivo é suportado"""
        return Path(file_path).suffix.lower() in cls.SUPPORTED_EXTENSIONS
    
    @staticmethod
    def read_txt(path: Union[str, Path]) -> str:
        """Lê o conteúdo de um arquivo .txt"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            if not content.strip() and Path(path).exists(): # Check exists to avoid warning on already logged FileNotFoundError
                logger.warning(f"Arquivo TXT vazio ou apenas com espaços em branco (UTF-8): {path}")
            return content
        except UnicodeDecodeError:
            logger.warning(f"Falha ao decodificar {path} com UTF-8. Tentando com Latin-1.")
            try:
                with open(path, 'r', encoding='latin-1') as f:
                   content = f.read()
                if not content.strip() and Path(path).exists():
                    logger.warning(f"Arquivo TXT vazio ou apenas com espaços em branco (Latin-1): {path}")
                return content
            except Exception as e_latin1:
                logger.error(f"Erro ao ler arquivo TXT {path} com Latin-1: {e_latin1}")
                return "" # Retorna string vazia em caso de erro na leitura com Latin-1
        except FileNotFoundError:
            logger.error(f"Arquivo TXT não encontrado: {path}")
            return "" # Retorna string vazia se o arquivo não for encontrado
        except Exception as e_general:
            logger.error(f"Erro inesperado ao ler arquivo TXT {path}: {e_general}")
            return "" # Retorna string vazia em outros erros de IO
# ... (previous code from your rag_gemini_improved.py snippet)
# ... (rest of DocumentProcessor, RAGSystem class, etc.)

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app) # If you use Cross-Origin Resource Sharing

# Optional: Initialize Flask-Limiter if you're using it
# limiter = Limiter(get_remote_address, app=app, default_limits=["200 per day", "50 per hour"])
# limiter.init_app(app) # Or directly pass app to Limiter constructor

# --- Global/Application-level Initializations ---
# It's good practice to initialize configurations and critical components once.
try:
    config = ConfigManager() # Loads and validates config from .env
    logger.info("Configuration loaded successfully.")

    # Configure Google Generative AI
    if config.google_api_key and config.google_api_key != "your_google_api_key_here":
        genai.configure(api_key=config.google_api_key)
        logger.info("Google Generative AI configured.")
    else:
        logger.error("GOOGLE_API_KEY is not set or is a placeholder. Gemini features will be unavailable.")
        # Depending on your app's logic, you might want to prevent startup
        # or operate in a degraded mode.

    # Initialize your RAG system, embedding models, FAISS index etc. here
    # e.g., embedding_model = SentenceTransformer(config.embedding_model_name)
    # e.g., faiss_index = # load or build your FAISS index
    # e.g., rag_system = RAGSystem(config, embedding_model, faiss_index)
    logger.info("Core RAG components initialized (placeholder).")

except ValueError as ve: # Catch ConfigManager validation errors specifically
    logger.critical(f"Configuration error: {ve}. Application cannot start.")
    # sys.exit(1) # Exit if config is invalid
except Exception as e:
    logger.critical(f"Fatal error during application initialization: {e}", exc_info=True)
    # sys.exit(1) # Exit on other critical init errors

# --- Flask Routes ---
@app.route('/')
def health_check():
    logger.debug("Health check endpoint '/' accessed.")
    return jsonify({"status": "RAG Gemini System is running!", "timestamp": datetime.now().isoformat()})

@app.route('/api/ask', methods=['POST'])
def ask_question_api():
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            logger.warning("API /api/ask: Bad request - 'question' field missing.")
            return jsonify({"error": "Missing 'question' in request body"}), 400
        
        question = data['question']
        logger.info(f"API /api/ask: Received question: '{question[:50]}...'") # Log snippet
        
        # --- Placeholder for your RAG logic ---
        # answer = rag_system.get_answer(question) # Assuming rag_system is initialized
        answer = f"This is a placeholder answer to your question: '{question}'"
        # --- End Placeholder ---
        
        return jsonify({"question": question, "answer": answer})
    except Exception as e:
        logger.error(f"API /api/ask: Error processing request: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500

# --- Entry Point Functions (for setup.py console_scripts) ---
def run_server():
    """Initializes and runs the Flask development server."""
    # Logger should already be configured by the global setup_logging() call.
    # Config and genai should already be initialized globally.
    # If GOOGLE_API_KEY was not set, a warning was already logged.
    # Consider if the server should even start if critical configs are missing.
    if not config.google_api_key or config.google_api_key == "your_google_api_key_here":
        logger.warning("Attempting to start server, but GOOGLE_API_KEY is not properly set.")
        # You might choose to not start the server:
        # print("ERROR: GOOGLE_API_KEY is not configured. Server will not start.")
        # return

    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", 5000))
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    
    logger.info(f"🚀 Starting Flask server on http://{host}:{port} (Debug: {debug_mode})")
    try:
        # Use 'app' (your Flask instance)
        app.run(host=host, port=port, debug=debug_mode, use_reloader=debug_mode)
    except Exception as e:
        logger.critical(f"Flask server failed to run: {e}", exc_info=True)

def main():
    """Main function to start the application (e.g., run the server)."""
    logger.info("🚀 Initializing RAG Gemini System via main()...")
    # Any other application-wide setup that main() should trigger can go here.
    run_server()

# --- Main Execution Block ---
if __name__ == '__main__':
    # The global logger is already set up at the beginning of the script.
    # The global config and genai configuration are also done.
    main()
        