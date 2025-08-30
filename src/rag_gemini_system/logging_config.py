# -*- coding: utf-8 -*-
"""
Módulo de Configuração de Logging.

Este arquivo centraliza a configuração do sistema de logging para garantir
consistência em toda a aplicação, seja executando via CLI ou como um servidor web.
"""

import os
import logging
import sys
from pathlib import Path

def setup_logging():
    """
    Configura o logging para a aplicação.

    Lê as configurações de variáveis de ambiente e estabelece o formato,
    o nível e os handlers (console e, opcionalmente, arquivo) para os logs.

    Esta função deve ser chamada uma única vez no ponto de entrada da aplicação.
    """
    # Evita reconfiguração se o logging já foi configurado
    if logging.getLogger().handlers:
        return logging.getLogger(__name__)

    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_file = os.getenv("LOG_FILE", "")
    # Formato de log mais detalhado
    log_format = "%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)] # Força a saída para stdout

    if log_file and log_file.strip():
        try:
            log_file_path = Path(log_file)
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            # Adiciona um FileHandler para escrever os logs em um arquivo
            handlers.append(logging.FileHandler(log_file_path, encoding='utf-8'))
        except Exception as e:
            # Usa um print aqui porque o logger ainda não está totalmente configurado
            print(f"ERRO CRÍTICO: Não foi possível configurar o logging para o arquivo {log_file}: {e}", file=sys.stderr)

    log_level = getattr(logging, log_level_str, logging.INFO)
    
    # Configura o logger raiz
    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)
    
    # Retorna um logger para o módulo que chamou a configuração
    return logging.getLogger(__name__)
