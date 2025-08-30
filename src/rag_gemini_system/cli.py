# -*- coding: utf-8 -*-
"""
Módulo de Interface de Linha de Comando (CLI) do Sistema RAG com Google Gemini.
Utiliza Typer para uma gestão de comandos robusta.
"""

# Importações de bibliotecas padrão
import os
import sys
import logging
import asyncio
from typing import Annotated
import typer


# Carrega variáveis de ambiente do arquivo .env
from dotenv import load_dotenv

# Importações dos módulos locais do sistema RAG
from .config import ConfigManager
from .rag_core import RAGSystem
from .logging_config import setup_logging

load_dotenv()

# Configura o logging e obtém uma instância do logger
setup_logging()
logger = logging.getLogger(__name__)

# Cria a aplicação Typer
app = typer.Typer(
    name="rag-gemini",
    help="Um sistema RAG com Google Gemini para responder perguntas com base em documentos.",
    add_completion=False
)

@app.command(name="process", help="Processa os documentos-fonte e constrói/atualiza o índice vetorial.")
def processar_documentos_cli():
    """
    Processa os documentos configurados em .env e os indexa no banco de vetores.
    """
    logger.info("CLI: Iniciando processamento de documentos...")
    try:
        config = ConfigManager()
        rag_system = RAGSystem(config=config)
        rag_system.initialize() # Correctly initialize the system
        rag_system.process_and_index_documents()
        logger.info(f"CLI: Processamento de '{config.target_document_name}' concluído com sucesso.")
        print(f"Documento '{config.target_document_name}' processado e indexado.")
    except Exception as e:
        logger.critical(f"Erro crítico durante o processamento de documentos: {e}", exc_info=True)
        print(f"Erro ao processar documentos. Verifique o arquivo de log para detalhes.")
        raise typer.Exit(code=1)

@app.command(name="ask", help="Faz uma pergunta ao sistema RAG com base nos documentos processados.")
def fazer_pergunta_cli(
    pergunta: Annotated[str, typer.Argument(..., help="O texto da pergunta a ser feita.")]
):
    """
    Envia uma pergunta para o sistema RAG e imprime a resposta em stream.
    """
    logger.info(f"CLI: Recebida pergunta: '{pergunta[:100]}'...")
    try:
        config = ConfigManager()
        rag_system = RAGSystem(config=config)
        rag_system.initialize() # Make sure the system is initialized

        if not rag_system.documents_processed:
            print(f"\n⚠️ Aviso: O documento '{rag_system.config.target_document_name}' ainda não foi processado.")
            print("Execute 'rag-gemini process' primeiro.")
            raise typer.Exit()

        print("\nBuscando fontes...")
        source_documents = rag_system.get_source_documents(pergunta)
        
        if source_documents:
            print(f"📄 Fontes encontradas com base em '{rag_system.config.target_document_name}':")
            for doc in source_documents:
                print(f"  - Trecho: \"{doc.page_content[:80]}\"...")
        else:
            print("  - Nenhuma fonte encontrada.")

        print("\nGerando resposta...\n")
        
        full_response = ""
        for token in rag_system.stream_answer(pergunta):
            print(token, end="", flush=True)
            full_response += token
        
        print("\n")
        logger.info("CLI: Pergunta respondida com sucesso.")

    except Exception as e:
        logger.critical(f"Erro crítico ao fazer pergunta: {e}", exc_info=True)
        print(f"❌ Erro ao buscar resposta. Verifique o arquivo de log para detalhes.")
        raise typer.Exit(code=1)



@app.command(name="server", help="Inicia o servidor web para a interface do chatbot.")
def run_server_cli(
    host: Annotated[str, typer.Option(help="O host para o servidor.")] = "0.0.0.0",
    port: Annotated[int, typer.Option(help="A porta para o servidor.")] = 5000,
    debug: Annotated[bool, typer.Option(help="Ativa o modo de depuração do Flask.")] = False,
):
    """
    Inicia o servidor web Flask com Waitress para produção ou o servidor de desenvolvimento.
    """
    logger.info("CLI: Iniciando o servidor web...")
    try:
        # Importações locais para o comando do servidor
        from .app import create_app
        from waitress import serve

        app_flask = create_app()
        
        # Usa as configurações do .env como padrão, mas permite override pela CLI
        config = ConfigManager()
        final_host = host if host != "0.0.0.0" else config.flask_host
        final_port = port if port != 5000 else config.flask_port
        final_debug = debug or os.getenv("FLASK_DEBUG", "False").lower() == "true"

        if final_debug:
            logger.info(f"Iniciando servidor de DESENVOLVIMENTO Flask em http://{final_host}:{final_port}")
            app_flask.run(host=final_host, port=final_port, debug=True, use_reloader=True)
        else:
            logger.info(f"Iniciando servidor de PRODUÇÃO Waitress em http://{final_host}:{final_port}")
            serve(app_flask, host=final_host, port=final_port)

    except ImportError as e:
        if "waitress" in str(e):
            logger.critical("Erro: A biblioteca 'waitress' não está instalada. Execute 'pip install waitress'.")
            print("Erro: 'waitress' não instalado. Execute 'pip install waitress' para rodar o servidor de produção.")
        else:
            logger.critical(f"Erro de importação não tratada: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.critical(f"Falha crítica ao iniciar o servidor: {e}", exc_info=True)
        print(f"Erro ao iniciar o servidor. Verifique o arquivo de log para detalhes.")
        raise typer.Exit(code=1)

def main():
    """
    Função de entrada para a aplicação Typer.
    """
    app()

if __name__ == "__main__":
    main()
