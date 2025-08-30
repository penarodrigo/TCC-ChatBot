# -*- coding: utf-8 -*-
"""
Módulo principal do sistema RAG, incluindo gerenciamento de cache e a lógica de RAG.
"""

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Dict, Any
from operator import itemgetter

from sentence_transformers import CrossEncoder

# LangChain Imports
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema.document import Document as LangChainDocument
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel

from google.cloud import texttospeech

from .config import ConfigManager
from .document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

class CrossEncoderReranker(BaseDocumentCompressor):
    """Compressor de documentos que usa um Cross-Encoder para re-classificar documentos."""
    model: CrossEncoder
    top_k: int = 3

    class Config:
        arbitrary_types_allowed = True

    def compress_documents(
        self, documents: Sequence[LangChainDocument], query: str, callbacks: Optional[Any] = None
    ) -> Sequence[LangChainDocument]:
        """Re-classifica documentos com base em uma consulta."""
        if not documents:
            return []
        
        doc_contents = [doc.page_content for doc in documents]
        pairs = [[query, doc_content] for doc_content in doc_contents]
        
        logger.info(f"Re-ranqueando {len(documents)} documentos com Cross-Encoder...")
        scores = self.model.predict(pairs)
        
        scored_docs = list(zip(scores, documents))
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        logger.info(f"Documentos re-ranqueados. Score do melhor: {scored_docs[0][0]:.4f}")
        return [doc for _, doc in scored_docs[:self.top_k]]

class RAGSystem:
    """Coordena todo o processo de RAG: processamento, indexação e geração de respostas."""
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = logging.getLogger(__name__ + ".RAGSystem")
        self.embedding_function: Optional[HuggingFaceEmbeddings] = None
        self.gemini_model: Optional[ChatGoogleGenerativeAI] = None
        self.vector_store: Optional[Chroma] = None
        self.retriever: Optional[ContextualCompressionRetriever] = None
        self.rag_chain: Optional[RunnableLambda] = None
        self.documents_processed = False

    def initialize(self):
        """Inicializa os modelos e componentes pesados e constrói a cadeia RAG."""
        self.logger.info("Inicializando RAGSystem...")

        self._initialize_embeddings()
        self._initialize_llm()
        
        self.process_and_index_documents()
        self._build_rag_chain()
        self.logger.info("RAGSystem inicializado e cadeia RAG construída com sucesso.")

    def _initialize_embeddings(self):
        """Carrega o modelo de embedding, aplicando cache se configurado."""
        try:
            # Importações com os caminhos corretos
            from langchain.storage import LocalFileStore
            from langchain.embeddings import CacheBackedEmbeddings

            self.logger.info(f"Carregando modelo de embedding base: {self.config.embedding_model_name}")
            
            # 1. Cria o modelo de embedding base
            underlying_embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model_name,
                model_kwargs={'device': 'cpu'} # Força CPU para compatibilidade
            )

            # 2. Verifica se o cache deve ser usado
            if self.config.enable_cache:
                self.logger.info("Cache de embeddings está ATIVADO.")
                
                # Cria o diretório de cache
                store = LocalFileStore(root_path=self.config.cache_dir)
                
                # Envolve o modelo de embedding com o cache
                cached_embedder = CacheBackedEmbeddings.from_bytes_store(
                    underlying_embeddings, store, namespace=self.config.embedding_model_name
                )
                
                self.embedding_function = cached_embedder
                self.logger.info(f"Usando CacheBackedEmbeddings com armazenamento em: '{self.config.cache_dir}'")
            else:
                self.logger.info("Cache de embeddings está DESATIVADO.")
                self.embedding_function = underlying_embeddings

            self.logger.info("Modelo de embedding pronto para uso.")

        except Exception as e:
            self.logger.error(f"Falha ao carregar ou configurar o modelo de embedding: {e}", exc_info=True)
            raise

    def _initialize_llm(self):
        """Configura o modelo de linguagem (LLM) Gemini."""
        if not self.config.google_api_key or self.config.google_api_key == "your_google_api_key_here":
            self.logger.warning("API Key do Gemini não configurada. Modelo não será inicializado.")
            return
            
        try:
            self.logger.info(f"Configurando modelo Gemini via LangChain: {self.config.gemini_model_name}")
            # Mapeamento de strings para os enums do LangChain
            safety_settings = {
                HarmCategory[key]: HarmBlockThreshold[value]
                for key, value in self.config.gemini_safety_settings.items()
            }

            self.gemini_model = ChatGoogleGenerativeAI(
                model=self.config.gemini_model_name,
                google_api_key=self.config.google_api_key,
                temperature=self.config.gemini_temperature,
                max_output_tokens=self.config.gemini_max_tokens,
                safety_settings=safety_settings,
                convert_system_message_to_human=True
            )
            self.logger.info("Modelo Gemini configurado via LangChain.")
        except Exception as e:
            self.logger.error(f"Falha ao configurar Gemini via LangChain: {e}", exc_info=True)
            raise

    def _get_text_chunks(self, text: str) -> List[str]:
        """Divide o texto em chunks com sobreposição."""
        if not text or not text.strip():
            return []
        # Lógica de chunking simples
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.config.chunk_size
            chunks.append(text[start:end])
            start += self.config.chunk_size - self.config.chunk_overlap
        return chunks

    def process_and_index_documents(self):
        """Processa e indexa o documento alvo se necessário."""
        if not self.embedding_function:
            self.logger.error("Modelo de embedding não carregado. Abortando processamento.")
            return

        doc_path = Path(self.config.data_dir) / self.config.target_document_name
        persist_directory = str(Path(self.config.vector_store_dir))

        self.logger.info(f"Verificando se o vector store já existe em: {persist_directory}")

        if Path(persist_directory).exists() and any(Path(persist_directory).iterdir()):
            self.logger.info("Carregando vector store existente...")
            self.vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embedding_function
            )
            self.documents_processed = True
            self.logger.info(f"Vector store carregado. Contém {self.vector_store._collection.count()} documentos.")
        else:
            self._create_new_vector_store(doc_path, persist_directory)

    def _create_new_vector_store(self, doc_path: Path, persist_directory: str):
        """Cria um novo vector store a partir de um documento."""
        self.logger.info(f"Criando novo vector store. Processando documento: {doc_path}")
        if not doc_path.is_file():
            self.logger.error(f"Documento alvo '{doc_path.name}' não encontrado em '{self.config.data_dir}'.")
            return

        text_content = DocumentProcessor.extract_text(doc_path)
        if not text_content:
            self.logger.warning(f"Nenhum texto extraído de {doc_path.name}. Verifique o arquivo.")
            return

        chunks = self._get_text_chunks(text_content)
        if not chunks:
            self.logger.warning(f"Nenhum chunk gerado para {doc_path.name}.")
            return

        lc_documents = [LangChainDocument(page_content=chunk) for chunk in chunks]

        self.logger.info(f"Gerando embeddings e indexando {len(lc_documents)} chunks...")
        try:
            self.vector_store = Chroma.from_documents(
                documents=lc_documents,
                embedding=self.embedding_function,
                persist_directory=persist_directory
            )
            self.documents_processed = True
            self.logger.info(f"Processamento de '{doc_path.name}' concluído. Vector store salvo em {persist_directory}")
        except Exception as e:
            self.logger.error(f"Erro ao criar e persistir o Chroma DB: {e}", exc_info=True)

    def _build_rag_chain(self):
        """Constrói a cadeia RAG completa usando LCEL de forma mais fluida."""
        if not self.vector_store or not self.gemini_model:
            self.logger.error("Vector store ou modelo Gemini não inicializados. Não é possível construir a cadeia RAG.")
            return

        # 1. Configurar o retriever (com ou sem re-ranking)
        self._setup_retriever()

        # 2. Definir o template do prompt
        template = """Você é um assistente focado em responder perguntas sobre o documento '{doc_name}'.
Use o contexto abaixo para basear sua resposta. Seja breve e direto.
Se o contexto não contiver a resposta, diga que a informação não foi encontrada no documento.

Contexto:
{context}

Pergunta: {question}

Resposta:"""
        prompt = ChatPromptTemplate.from_template(template)

        # 3. Função para formatar os documentos recuperados
        def format_docs(docs: Sequence[LangChainDocument]) -> str:
            if not docs:
                return f"Nenhum contexto relevante encontrado no documento '{self.config.target_document_name}'."
            return "\n\n---\n\n".join(doc.page_content for doc in docs)

        # 4. Construir a cadeia RAG completa com LCEL
        rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
            | prompt
            | self.gemini_model
            | StrOutputParser()
        )

        self.rag_chain = (
            RunnablePassthrough.assign(
                context=itemgetter("question") | self.retriever,
                doc_name=lambda x: self.config.target_document_name
            ).assign(answer=rag_chain_from_docs)
        )
        
        self.logger.info("Cadeia RAG completa construída com sucesso.")

    def _setup_retriever(self):
        """Configura o retriever, aplicando o re-ranker se disponível."""
        base_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.config.max_context_docs * 3}
        )

        # Desativando o re-ranking para melhorar a performance
        self.retriever = base_retriever
        self.logger.info("Retriever configurado SEM re-ranking (para performance).")

    def get_source_documents(self, question: str) -> List[LangChainDocument]:
        """
        Recupera os documentos fonte para uma dada pergunta.
        """
        if not self.retriever:
            self.logger.error("Retriever não inicializado.")
            return []
        return self.retriever.invoke(question)

    def stream_answer(self, question: str):
        """
        Invoca a cadeia RAG com streaming e gera os tokens da resposta.
        """
        if not self.rag_chain:
            yield "Erro: A cadeia RAG não foi inicializada corretamente."
            return
        if not self.documents_processed:
            yield f"Erro: O documento alvo '{self.config.target_document_name}' ainda não foi processado."
            return

        self.logger.info(f"Iniciando stream da cadeia RAG para a pergunta: '{question[:50]}...'")
        
        # O stream vai gerar dicionários, e queremos o valor da chave 'answer'
        for chunk in self.rag_chain.stream({"question": question}):
            answer_token = chunk.get('answer')
            if answer_token:
                yield answer_token

    def get_answer(self, question: str) -> Dict[str, Any]:
        """
        Coleta uma resposta completa a partir do stream para manter a compatibilidade.
        """
        full_answer = ""
        source_documents = self.get_source_documents(question)
        
        for token in self.stream_answer(question):
            full_answer += token

        return {"answer": full_answer, "source_documents": source_documents}

    def synthesize_speech(self, text: str) -> Optional[bytes]:
        """
        Sintetiza a fala a partir do texto usando o Google Cloud Text-to-Speech.

        Args:
            text: O texto a ser convertido em fala.

        Returns:
            Os bytes do conteúdo de áudio ou None se ocorrer um erro.
        """
        try:
            self.logger.info(f"Iniciando síntese de fala para o texto: '{text[:50]}...'")
            
            client_options = {
                "api_endpoint": "texttospeech.googleapis.com",
                "quota_project_id": self.config.quota_project_id
            }

            # Instancia o cliente. A autenticação é tratada automaticamente
            # pelo ambiente (gcloud auth application-default login).
            client = texttospeech.TextToSpeechClient(client_options=client_options)

            synthesis_input = texttospeech.SynthesisInput(text=text)

            # Configura a voz (pode ser personalizada via config)
            voice = texttospeech.VoiceSelectionParams(
                language_code="pt-BR",
                name="pt-BR-Standard-B",  # Voz feminina
                ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
            )

            # Configura o formato do áudio
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )

            response = client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )

            self.logger.info("Síntese de fala concluída com sucesso.")
            return response.audio_content

        except Exception as e:
            self.logger.error(f"Falha na síntese de fala: {e}", exc_info=True)
            # Adicione um log mais detalhado sobre a autenticação
            if "Could not automatically determine credentials" in str(e):
                self.logger.error(
                    "Erro de autenticação com a API Google Text-to-Speech. "
                    "Verifique se você executou 'gcloud auth application-default login' "
                    "ou se as credenciais estão configuradas no ambiente."
                )
            return None

