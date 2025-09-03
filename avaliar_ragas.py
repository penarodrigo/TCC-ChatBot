# evaluate_ragas.py
import os
import json
import sys
import re
import time
import argparse
import traceback
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    answer_correctness,
)
from langchain_google_genai import ChatGoogleGenerativeAI
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
import ragas
import google.api_core.exceptions

from src.rag_gemini_system.config import ConfigManager
from src.rag_gemini_system.rag_core import RAGSystem

def load_qa_from_file(file_path: str) -> tuple[list[str], list[str]]:
    """
    Carrega perguntas e respostas de um arquivo de texto formatado.
    """
    questions, ground_truths = [], []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        pattern = re.compile(r'^\d+\.\s*(.*?)\nResposta:\s*(.*?)$', re.MULTILINE)
        matches = pattern.findall(content)
        if matches:
            questions, ground_truths = [list(t) for t in zip(*matches)]
    except FileNotFoundError:
        print(f"Erro: Arquivo de perguntas e respostas '{file_path}' não encontrado.")
    except Exception as e:
        print(f"Erro ao ler ou processar o arquivo '{file_path}': {e}")
    return questions, ground_truths

def create_rag_system():
    """
    Cria e inicializa o sistema RAG.
    """
    try:
        config = ConfigManager()
        rag_system = RAGSystem(config=config)
        rag_system.initialize()  # Inicializa os modelos e processa os documentos
        
        if not rag_system.documents_processed:
            print(f"Falha ao processar o documento '{config.target_document_name}' durante a inicialização.")
            return None
            
        return rag_system
    except Exception as e:
        print(f"Erro ao inicializar o sistema RAG: {e}")
        traceback.print_exc()
        return None

def run_ragas_evaluation(qa_file: str, output_file: str, batch_size: int = 0, batch_number: int = 0, delay: int = 6):
    """
    Executa a avaliação do RAG com RAGAS, com suporte a processamento em lotes.
    """
    start_time = time.perf_counter()

    config = ConfigManager()
    rag_system = create_rag_system()
    if not rag_system:
        return

    gemini_api_key = config.google_api_key
    if not gemini_api_key or gemini_api_key == "your_google_api_key_here":
        print("Chave da API do Google (GOOGLE_API_KEY) não configurada. Saindo.")
        return

    llm_judge = ChatGoogleGenerativeAI(
        model=config.gemini_model_name,
        google_api_key=gemini_api_key,
        temperature=0.0
    )

    try:
        ragas_embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model_name,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        ragas.embeddings = ragas_embeddings
        print(f"Embeddings para RAGAS configurados com: {config.embedding_model_name}")
    except Exception as e:
        print(f"Aviso: Falha ao configurar embeddings customizados para RAGAS: {e}. RAGAS pode usar defaults.")
        ragas_embeddings = None

    print(f"Carregando perguntas e respostas de '{qa_file}'...")
    all_questions, all_ground_truths = load_qa_from_file(qa_file)

    if not all_questions:
        print("Nenhum par de pergunta/resposta foi carregado. Encerrando.")
        return

    # Lógica de processamento em lotes
    if batch_size > 0 and batch_number > 0:
        start_index = (batch_number - 1) * batch_size
        end_index = start_index + batch_size
        evaluation_questions = all_questions[start_index:end_index]
        ground_truths = all_ground_truths[start_index:end_index]
        
        if not evaluation_questions:
            print(f"Lote {batch_number} com tamanho {batch_size} está vazio. Nada para processar.")
            return
            
        print(f"Processando Lote {batch_number}: {len(evaluation_questions)} de {len(all_questions)} perguntas (índices {start_index} a {end_index-1}).")
        
        # Modifica o nome do arquivo de saída para refletir o lote
        base, ext = os.path.splitext(output_file)
        output_file = f"{base}_batch_{batch_number}{ext}"
    else:
        evaluation_questions = all_questions
        ground_truths = all_ground_truths
        print(f"{len(evaluation_questions)} pares de pergunta/resposta carregados (processando todos).")

    data_samples_list = []
    print("Coletando dados para avaliação RAGAS...")
    for i, q in enumerate(evaluation_questions):
        print(f"  Processando pergunta {i+1}/{len(evaluation_questions)}: \"{q}\"")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                generated_answer = rag_system.get_answer(q)['answer']
                retrieved_contexts = rag_system.get_last_retrieved_contexts()
                if delay > 0:
                    time.sleep(delay) # Pausa para evitar limite de taxa
                break
            except google.api_core.exceptions.ResourceExhausted as e:
                if "per day" in str(e).lower():
                    print(f"ERRO CRÍTICO: Cota diária da API atingida. {e}")
                    sys.exit(1)
                wait_time = (attempt + 1) * 20
                print(f"    ATINGIU LIMITE DE TAXA. Tentativa {attempt + 1}/{max_retries}. Aguardando {wait_time}s.")
                time.sleep(wait_time)
            except Exception as e:
                print(f"    ERRO INESPERADO: {e}. Pulando pergunta.")
                generated_answer = "Erro"
                retrieved_contexts = []
                break
        else:
            print(f"    FALHA APÓS {max_retries} TENTATIVAS. Pulando pergunta.")
            generated_answer = "Falha"
            retrieved_contexts = []

        data_samples_list.append({
            "question": q,
            "answer": generated_answer,
            "contexts": retrieved_contexts or ["Nenhum contexto recuperado."],
            "ground_truth": ground_truths[i]
        })

    if not data_samples_list:
        print("Nenhum dado de avaliação gerado.")
        return

    dataset = Dataset.from_list(data_samples_list)
    print(f"Dataset de avaliação criado com {len(dataset)} amostras.")

    metrics_to_evaluate = [
        faithfulness,
        answer_relevancy,
        context_precision,
        answer_correctness,
    ]
    
    print("Iniciando avaliação com RAGAS...")
    try:
        results = evaluate(
            dataset,
            metrics=metrics_to_evaluate,
            llm=llm_judge,
            embeddings=ragas_embeddings,
        )
        print("\nResultados da Avaliação RAGAS:")
        print(results)
        
        results_df = results.to_pandas()
        print("\nResultados da Avaliação RAGAS (DataFrame):")
        print(results_df)

        results_dict_for_json = results_df.to_dict(orient='records')
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results_dict_for_json, f, indent=4, ensure_ascii=False)
        print(f"\nResultados salvos em {output_file}")

    except Exception as e:
        print(f"Erro durante a avaliação RAGAS: {e}")
        traceback.print_exc()

    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f"\n--- Relatório de Tempo ---")
    print(f"Tempo total de execução: {total_time:.2f} segundos")
    print(f"--------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avalia o sistema RAG com RAGAS.")
    parser.add_argument("--qa_file", type=str, default="perguntas_respostas_IDDC.txt", help="Caminho para o arquivo de perguntas e respostas.")
    parser.add_argument("--output_file", type=str, default="ragas_evaluation_results.json", help="Caminho para o arquivo de saída dos resultados.")
    parser.add_argument("--batch_size", type=int, default=0, help="Tamanho do lote para avaliação. Se 0, processa tudo.")
    parser.add_argument("--batch_number", type=int, default=0, help="Número do lote a ser processado (começa em 1).")
    parser.add_argument("--delay", type=int, default=6, help="Pausa em segundos entre as perguntas para controlar a taxa de requisições.")
    args = parser.parse_args()
    
    run_ragas_evaluation(args.qa_file, args.output_file, args.batch_size, args.batch_number, args.delay)