# evaluate_ragas.py
import os
import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    # context_recall, # Removido pois requer 'ground_truth_contexts' que não temos neste exemplo simples
    answer_correctness,
)
# Para RAGAS > 0.1.0, você passa a instância do LLM Langchain diretamente
from langchain_google_genai import ChatGoogleGenerativeAI
# Importar FAISS para detecção de GPU, se aplicável
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
# Se for usar embeddings customizados para RAGAS (ex: para answer_relevancy)
from langchain_community.embeddings import HuggingFaceEmbeddings
import ragas # Para configurar embeddings globalmente, se necessário

# Importe seu RAGSystem e ConfigManager
# A inicialização global em nucleo_rag.py será acionada aqui.
from nucleo_rag import rag_system_instance, config

def run_ragas_evaluation():
    # 1. Verificação da inicialização (initialize_global_components é chamado na importação)
    if not rag_system_instance or not config:
        print("Erro: Sistema RAG ou configuração não inicializados. Verifique nucleo_rag.py.")
        return

    # 2. Preparar LLM Juiz para RAGAS
    gemini_api_key = config.google_api_key
    if not gemini_api_key or gemini_api_key == "your_google_api_key_here":
        print("Chave da API do Google (GOOGLE_API_KEY) não configurada para o LLM juiz do RAGAS. Saindo.")
        return

    llm_judge = ChatGoogleGenerativeAI(
        model=config.gemini_model_name,  # Pode ser o mesmo modelo do RAG ou um diferente
        google_api_key=gemini_api_key,
        temperature=0.0  # Para julgamento, geralmente queremos respostas determinísticas
    )

    # Configurar embeddings para RAGAS (especialmente para answer_relevancy)
    # Se não configurado, RAGAS pode usar embeddings da OpenAI por padrão.
    # Para consistência, use os mesmos embeddings do seu sistema RAG.
    try:
        ragas_embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model_name,
            model_kwargs={'device': 'cuda' if FAISS_AVAILABLE and faiss and faiss.get_num_gpus() > 0 else 'cpu'},
            encode_kwargs={'normalize_embeddings': True} # E5 models often benefit
        )
        # Algumas versões/métricas do RAGAS podem pegar de ragas.embeddings
        ragas.embeddings = ragas_embeddings 
        print(f"Embeddings para RAGAS configurados com: {config.embedding_model_name}")
    except Exception as e:
        print(f"Aviso: Falha ao configurar embeddings customizados para RAGAS: {e}. RAGAS pode usar defaults.")


    # 3. Coletar/Carregar seu Dataset de Avaliação
    # Este é um exemplo. Você precisará criar este dataset com perguntas relevantes ao IDDC.pdf.
    evaluation_questions = [
        "Qual o objetivo principal do IDDC?",
        "Quais são as diretrizes para a classificação da informação?",
        "Quem é responsável por garantir a segurança da informação na CEMIG segundo o IDDC?",
        "O que o IDDC diz sobre o uso de dispositivos móveis pessoais?"
    ]
    ground_truths = [ # Respostas ideais, escritas por um humano, baseadas no IDDC.pdf
        "O objetivo principal do IDDC é estabelecer diretrizes e responsabilidades para proteger os ativos de informação da CEMIG contra ameaças, garantindo sua confidencialidade, integridade e disponibilidade.",
        "O IDDC estabelece que a informação deve ser classificada em níveis como Pública, Interna, Confidencial e Restrita, de acordo com sua sensibilidade e impacto para o negócio.",
        "Segundo o IDDC, a responsabilidade pela segurança da informação é compartilhada, mas a alta administração, gestores de áreas e todos os colaboradores têm papéis específicos na sua proteção.",
        "O IDDC provavelmente estipula que o uso de dispositivos móveis pessoais para acessar informações corporativas deve seguir políticas específicas de segurança, como uso de senhas fortes, criptografia e instalação de softwares de segurança aprovados."
    ]

    if not rag_system_instance.documents_processed:
        print(f"Documento '{config.target_document_name}' não processado. Tentando processar agora...")
        rag_system_instance.process_and_index_documents() # Tenta processar
        if not rag_system_instance.documents_processed:
            print(f"Falha ao processar '{config.target_document_name}'. Saindo da avaliação.")
            return
    
    print(f"Sistema RAG focado no documento: {config.target_document_name}")

    data_samples_list = []
    print("Coletando dados para avaliação RAGAS...")
    for i, q in enumerate(evaluation_questions):
        print(f"  Processando pergunta para RAGAS: \"{q}\"")
        generated_answer = rag_system_instance.get_answer(q)
        retrieved_contexts = rag_system_instance.get_last_retrieved_contexts()
        
        if not generated_answer or not generated_answer.strip():
            print(f"    AVISO: Resposta gerada vazia para a pergunta: '{q}'. Usando placeholder.")
            generated_answer = "Não foi possível gerar uma resposta."
        if not retrieved_contexts: # retrieved_contexts é List[str]
            print(f"    AVISO: Contextos recuperados vazios para a pergunta: '{q}'. Usando placeholder.")
            retrieved_contexts = ["Nenhum contexto recuperado."]

        data_samples_list.append({
            "question": q,
            "answer": generated_answer,
            "contexts": retrieved_contexts, 
            "ground_truth": ground_truths[i] 
        })

    if not data_samples_list:
        print("Nenhum dado de avaliação gerado. Verifique as perguntas e o sistema RAG.")
        return

    dataset = Dataset.from_list(data_samples_list)
    print(f"Dataset de avaliação criado com {len(dataset)} amostras.")

    # 4. Definir Métricas e Avaliar
    # Para RAGAS > 0.1.0, injete o LLM nas métricas que o requerem.
    # Especifique embeddings para métricas que os utilizam, como answer_relevancy.
    metrics_to_evaluate = [
        faithfulness.inject(llm=llm_judge),
        answer_relevancy.inject(embeddings=ragas_embeddings if 'ragas_embeddings' in locals() else None),
        context_precision,
        # context_recall, # Requer 'ground_truth_contexts' (lista de contextos relevantes esperados)
        answer_correctness.inject(llm=llm_judge),
    ]
    
    print("Iniciando avaliação com RAGAS...")
    try:
        results = evaluate(
            dataset,
            metrics=metrics_to_evaluate,
            # llm=llm_judge, # Pode ser passado aqui se as métricas não tiverem .inject() ou para um fallback
        )
        print("\nResultados da Avaliação RAGAS (Objeto Dataset do Hugging Face):")
        print(results)
        
        results_df = results.to_pandas()
        print("\nResultados da Avaliação RAGAS (DataFrame):")
        print(results_df)

        # Salvar resultados em um arquivo JSON
        # Convertendo o DataFrame para um formato serializável em JSON
        results_dict_for_json = results_df.to_dict(orient='records') 
        
        output_file = "ragas_evaluation_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results_dict_for_json, f, indent=4, ensure_ascii=False)
        print(f"\nResultados salvos em {output_file}")

    except Exception as e:
        print(f"Erro durante a avaliação RAGAS: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # A inicialização dos componentes globais (config, rag_system_instance)
    # ocorre quando 'rag_gemini_improved' é importado.
    # load_dotenv já é chamado em rag_gemini_improved.py
    run_ragas_evaluation()
