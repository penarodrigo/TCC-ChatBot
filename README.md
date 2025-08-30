# 🚀 Sistema RAG com Google Gemini

Um sistema de **Retrieval-Augmented Generation (RAG)** robusto e modular, integrado com a API **Google Gemini**. Ele é projetado para processar um documento específico e responder perguntas com base em seu conteúdo, utilizando uma interface de chat moderna e uma API RESTful.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com)
[![Google Gemini](https://img.shields.io/badge/Google-Gemini-orange.svg)](https://ai.google.dev)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/penarodrigo/rag-gemini-system/blob/main/LICENSE)

## 📋 Índice

- [Sobre o Projeto](#-sobre-o-projeto)
- [Características](#-características)
- [Tecnologias](#-tecnologias)
- [Instalação](#-instalação)
- [Configuração](#-configuração)
- [Como Usar](#-como-usar)
- [API Endpoints](#-api-endpoints)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Avaliação de Performance](#-avaliação-de-performance)
- [Licença](#-licença)

## 🎯 Sobre o Projeto

Este projeto foi desenvolvido como uma solução de chatbot focada em responder perguntas sobre um documento específico (por exemplo, um manual técnico, um regulamento ou um relatório extenso). Ele extrai o texto, o divide em partes gerenciáveis (*chunks*), gera embeddings vetoriais e os armazena em um banco de dados vetorial **ChromaDB** para recuperação rápida e eficiente.

## ✨ Características

- **Processamento de Múltiplos Formatos**: Suporte nativo para `.pdf`, `.docx`, `.pptx`, `.xlsx`, `.csv` e `.txt`.
- **Interface de Chat Moderna**: Uma interface web responsiva com streaming de respostas em tempo real para uma experiência de usuário fluida.
- **Síntese de Voz (Text-to-Speech)**: Converte as respostas do chatbot em áudio com a tecnologia do Google Cloud Text-to-Speech.
- **Sistema de Feedback**: Permite que os usuários avaliem as respostas (like/dislike), fornecendo dados para futuras melhorias.
- **Busca Vetorial com ChromaDB**: Utiliza o ChromaDB para buscas de similaridade eficientes, garantindo respostas relevantes.
- **Cache de Embeddings**: Salva os embeddings gerados para evitar reprocessamento, economizando tempo e recursos da API.
- **Otimização de Tokens**: Configurações ajustadas para reduzir o consumo de tokens e otimizar os custos da API.
- **API RESTful**: Endpoints claros para integração, incluindo streaming de respostas e síntese de voz.
- **Comandos CLI**: Ferramentas de linha de comando para interações rápidas e gerenciamento do sistema.
- **Configuração Flexível**: Gerenciamento de configurações via arquivo `.env` para fácil customização.
- **Avaliação de Performance**: Inclui um script (`avaliar_ragas.py`) para medir a qualidade do sistema RAG.

## 🛠 Tecnologias

### Core
- **Python 3.8+**
- **Flask** - Framework web
- **Google Gemini** - Modelo de linguagem
- **Google Cloud Text-to-Speech** - Síntese de voz
- **Sentence-Transformers** - Geração de embeddings de texto
- **ChromaDB** - Banco de dados vetorial

### Processamento de Documentos
- **pdfplumber** - Extração de texto de PDFs
- **python-docx** - Processamento de Word
- **python-pptx** - Processamento de PowerPoint
- **openpyxl** - Processamento de Excel
- **pandas** - Manipulação de dados CSV e Excel

### Outros
- **python-dotenv** - Gerenciamento de configurações
- **numpy** - Operações numéricas
- **RAGAS** - Framework de avaliação de sistemas RAG

## 📦 Instalação

### Pré-requisitos
- Python 3.8 a 3.12 (Python 3.13 pode apresentar problemas de compatibilidade com algumas bibliotecas)
- Git

### Instalação Rápida

```bash
# 1. Clone o repositório
git clone https://github.com/penarodrigo/rag-gemini-system.git
cd TCC-ChatBot-LIMPO

# 2. Crie um ambiente virtual
python -m venv venv

# 3. Ative o ambiente virtual
# No Windows:
venv\Scripts\activate     # Windows
# No Linux/macOS:
# source venv/bin/activate

# 4. Instale as dependências
pip install -r requirements.txt

# 5. Instale o projeto em modo editável (para que as mudanças no código sejam refletidas)
pip install -e .

# Opcional: Instale as dependências de desenvolvimento e avaliação
pip install -r dev-requirements.txt
```

## ⚙️ Configuração

As configurações do projeto são gerenciadas através de um arquivo `.env`. Para começar, crie um novo arquivo chamado `.env` na raiz do projeto e preencha as seguintes variáveis:

```bash
# Chave da API do Google Gemini
GOOGLE_API_KEY="sua_chave_de_api_aqui"

# (Opcional) ID do Projeto Google Cloud para cotas do Text-to-Speech
QUOTA_PROJECT_ID="seu_project_id_aqui"

# (Opcional) Documento alvo a ser processado na pasta 'dados_rag'
TARGET_DOCUMENT="IDDC.pdf"

# (Opcional) Modelos de embedding e LLM
EMBEDDING_MODEL="intfloat/multilingual-e5-large"
GEMINI_MODEL="gemini-pro"
```

**Importante:**
- A `GOOGLE_API_KEY` é essencial para a comunicação com a API do Gemini.
- A `QUOTA_PROJECT_ID` é necessária se você estiver usando as credenciais padrão da aplicação (`gcloud auth application-default login`) para a funcionalidade de Text-to-Speech.

## 🌐 API Endpoints

A aplicação expõe os seguintes endpoints:

- `GET /health`: Verifica o status da aplicação.
- `GET /api/ask?question=<sua_pergunta>`: Envia uma pergunta e recebe a resposta via Server-Sent Events (SSE) para streaming em tempo real.
- `POST /api/synthesize`: Envia um texto no corpo da requisição (`{"text": "seu_texto"}`) e retorna o áudio correspondente em formato MP3.
- `POST /api/feedback`: Envia um feedback sobre uma resposta (like/dislike).
- `GET /api/history`: Retorna o histórico da conversa.
