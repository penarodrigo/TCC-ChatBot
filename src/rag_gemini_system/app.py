# -*- coding: utf-8 -*-
"""
Módulo para a aplicação web Flask do Sistema RAG com Google Gemini.
"""

import os
import sys
import logging
import csv
from pathlib import Path
from datetime import datetime
import json
import threading
import uuid

from flask import Flask, request, jsonify, render_template, g, Response
from flask_cors import CORS
from dotenv import load_dotenv

from rag_gemini_system.config import ConfigManager
from rag_gemini_system.rag_core import RAGSystem
from rag_gemini_system.logging_config import setup_logging

load_dotenv()

# Configura o logging assim que o módulo é carregado.
# Isso garante que qualquer log subsequente, incluindo o do Flask,
# use a configuração definida.
setup_logging()
logger = logging.getLogger(__name__)

feedback_lock = threading.Lock()
conversation_history = [] # Global list to store conversation history

def get_rag_system():
    """
    Retorna a instância do RAGSystem, inicializando-a se necessário.
    """
    if 'rag_system' not in g:
        logger.info("Inicializando RAGSystem no contexto da aplicação.")
        config = ConfigManager()
        g.rag_system = RAGSystem(config=config)
        g.rag_system.initialize()
    return g.rag_system

def create_app():
    """
    Cria e configura a aplicação Flask.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    app = Flask(__name__, 
                template_folder=os.path.join(project_root, 'templates'),
                static_folder=os.path.join(project_root, 'static'))
    CORS(app)

    @app.route('/health')
    def rota_principal_health_check():
        logger.debug("Endpoint '/health' acessado.")
        rag_system = get_rag_system()
        doc_name = rag_system.config.target_document_name
        return jsonify({"status": "Online!", "timestamp": datetime.now().isoformat(), "message": f"API RAG Gemini (foco: {doc_name})."})

    @app.route('/api/ask', methods=['GET'])
    def rota_ask_question_api():
        pergunta = request.args.get('question')
        if not pergunta or not isinstance(pergunta, str) or not pergunta.strip():
            return jsonify({"error": "'question' como query parameter é obrigatório."}), 400

        rag_system = get_rag_system()

        if not rag_system.documents_processed:
            return jsonify({"error": f"'{rag_system.config.target_document_name}' precisa ser processado."}) , 400

        def generate_stream():
            # 1. Get source documents
            source_documents = rag_system.get_source_documents(pergunta)
            source_docs_json = [
                {"page_content": doc.page_content, "metadata": doc.metadata} 
                for doc in source_documents
            ]
            yield f"event: source_documents\ndata: {json.dumps(source_docs_json)}\n\n"

            # 2. Stream the answer tokens
            full_answer_list = []
            for token in rag_system.stream_answer(pergunta):
                full_answer_list.append(token)
                yield f"event: answer_token\ndata: {json.dumps({'token': token})}\n\n"
            
            # 3. Signal the end of the stream
            yield "event: stream_end\ndata: {}\n\n"

            # After stream ends, add to conversation history
            full_answer = "".join(full_answer_list)
            conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "question": pergunta,
                "answer": full_answer
            })
            logger.info(f"Conversa adicionada ao histórico. Pergunta: '{pergunta[:50]}...', Resposta: '{full_answer[:50]}...'")

        return Response(generate_stream(), mimetype='text/event-stream')

    @app.route('/api/history', methods=['GET'])
    def rota_get_history_api():
        logger.debug("Endpoint '/api/history' acessado.")
        return jsonify({"history": conversation_history}), 200

    @app.route('/api/feedback', methods=['POST'])
    def rota_feedback_api():
        try:
            data = request.get_json()
            if not data or not all(k in data for k in ['question', 'answer', 'rating']):
                return jsonify({"error": "Dados de feedback incompletos. 'question', 'answer' e 'rating' são obrigatórios."}) , 400

            # Mapping: like (1) -> 5, dislike (0) -> 1
            score = 5 if data['rating'] == '1' else 1

            rag_system = get_rag_system()
            feedback_file_path = Path(rag_system.config.data_dir) / "feedback_likert.csv"

            with feedback_lock:
                file_exists = feedback_file_path.is_file()
                with open(feedback_file_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(['timestamp', 'question', 'answer', 'score'])
                    
                    timestamp = datetime.now().isoformat()
                    writer.writerow([timestamp, data['question'], data['answer'], score])
            
            logger.info(f"Feedback recebido e salvo para a pergunta: '{data['question'][:50]}...' com score {score}")
            return jsonify({"status": "success", "message": "Feedback recebido com sucesso."}) , 200
        except Exception as e:
            logger.error(f"API /api/feedback erro: {e}", exc_info=True)
            return jsonify({"error": "Erro interno ao processar feedback."}) , 500

    @app.route('/api/synthesize', methods=['POST'])
    def rota_synthesize_api():
        data = request.get_json()
        text = data.get('text')

        if not text or not isinstance(text, str) or not text.strip():
            return jsonify({'error': "O campo 'text' é obrigatório."}), 400

        rag_system = get_rag_system()
        audio_content = rag_system.synthesize_speech(text)

        if audio_content:
            return Response(audio_content, mimetype='audio/mpeg')
        else:
            return jsonify({'error': 'Falha ao gerar o áudio.'}), 500

    @app.route('/api/transcribe', methods=['POST'])
    def rota_transcribe_api():
        if 'file' not in request.files:
            return jsonify({"error": "Nenhum arquivo de áudio enviado."}), 400
        
        audio_file = request.files['file']
        audio_data = audio_file.read()

        rag_system = get_rag_system()
        transcript = rag_system.transcribe_audio(audio_data)

        if transcript:
            return jsonify({"transcript": transcript})
        else:
            return jsonify({"error": "Falha ao transcrever o áudio."}), 500

    @app.route('/', methods=['GET'])
    def interface_usuario_web():
        logger.debug("Acessando /")
        try:
            return render_template('index.html', title="ChatBot COD")
        except Exception as e:
            logger.error(f"Erro renderizando template para /: {e}", exc_info=True)
            return "Erro ao carregar interface. Verifique logs.", 500
            
    return app

# Create the app instance at the module level for Uvicorn
app = create_app()

def main():
    config = ConfigManager()
    host, port = config.flask_host, config.flask_port
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    logger.info(f"🚀 Iniciando Flask em http://{host}:{port} (Debug: {debug})")
    # app.run(host=host, port=port, debug=debug, use_reloader=debug) # Removed for Uvicorn

if __name__ == '__main__':
    # Only run main() if the script is executed directly (not imported by Uvicorn)
    main()
