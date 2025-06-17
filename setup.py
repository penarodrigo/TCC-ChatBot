#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for RAG System with Google Gemini
Sistema RAG com Google Gemini para VSCode
"""

from setuptools import setup, find_packages
import os
import sys

# Verificar versão do Python
if sys.version_info < (3, 8):
    sys.exit('Python 3.8 ou superior é necessário.')

# Ler o arquivo README
def read_readme():
    """Lê o arquivo README para a descrição longa"""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Sistema RAG (Retrieval-Augmented Generation) com Google Gemini"

# Ler os requirements
def read_requirements():
    """Lê o arquivo requirements.txt"""
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            # Filtrar comentários e linhas vazias
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

# Configurações do setup
setup(
    # Informações básicas do pacote
    name="rag-gemini-system",
    version="1.0.0",
    description="Sistema RAG (Retrieval-Augmented Generation) com Google Gemini",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    
    # Informações do autor
    author="penarodrigo",
    author_email="rodrigo.pena@cemig.com.br",
    
    # URLs do projeto
    url="https://github.com/penarodrigo/rag-gemini-system",
    project_urls={
        "Bug Reports": "https://github.com/penarodrigo/rag-gemini-system/issues",
        "Source": "https://github.com/penarodrigo/rag-gemini-system",
        "Documentation": "https://github.com/penarodrigo/rag-gemini-system/wiki",
    },
    
    # Classificações do projeto
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Framework :: Flask",
        "Natural Language :: Portuguese (Brazilian)",
        "Natural Language :: English",
    ],
    
    # Configurações de pacotes
    packages=find_packages(),
    include_package_data=True,
    
    # Versão mínima do Python
    python_requires='>=3.8',
    
    # Dependências
    install_requires=read_requirements(),
    
    # Dependências extras
    extras_require={
        'dev': [
            'pytest>=7.4.3',
            'pytest-flask>=1.3.0',
            'pytest-cov>=4.1.0',
            'black>=23.9.1',
            'flake8>=6.1.0',
            'mypy>=1.6.1',
        ],
        'gpu': [
            'faiss-gpu>=1.7.4',
        ],
        'docs': [
            'sphinx>=7.2.6',
            'sphinx-rtd-theme>=1.3.0',
        ],
        'all': [
            'pytest>=7.4.3',
            'pytest-flask>=1.3.0',
            'pytest-cov>=4.1.0',
            'black>=23.9.1',
            'flake8>=6.1.0',
            'mypy>=1.6.1',
            'faiss-gpu>=1.7.4',
            'sphinx>=7.2.6',
            'sphinx-rtd-theme>=1.3.0',
        ]
    },
    
    # Scripts de linha de comando
    entry_points={
        'console_scripts': [
            'rag-gemini=rag_gemini_improved:main',
            'rag-server=rag_gemini_improved:run_server',
        ],
    },
    
    # Arquivos de dados
    package_data={
        '': ['*.txt', '*.md', '*.yml', '*.yaml', '*.json'],
        'templates': ['*.html'],
        'static': ['*.css', '*.js', '*.png', '*.jpg', '*.gif'],
    },
    
    # Palavras-chave
    keywords=[
        'rag',
        'retrieval-augmented-generation',
        'google-gemini',
        'ai',
        'nlp',
        'machine-learning',
        'embeddings',
        'document-processing',
        'question-answering',
        'chatbot',
        'cemig',
        'energy',
        'flask',
        'api',
    ],
    
    # Licença
    license='MIT',
    
    # Configurações do zip
    zip_safe=False,
    
    # Plataformas suportadas
    platforms=['any'],
    
)

# Funções auxiliares para desenvolvimento
def run_tests():
    """Executa os testes"""
    import pytest
    return pytest.main(['-v', 'tests/'])

def run_linting():
    """Executa o linting do código"""
    import subprocess
    commands = [
        ['black', '--check', '.'],
        ['flake8', '.'],
        ['mypy', '.'],
    ]
    
    for cmd in commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f" {' '.join(cmd)} falhou:")
                print(result.stdout)
                print(result.stderr)
                return False
            else:
                print(f" {' '.join(cmd)} passou")
        except FileNotFoundError:
            print(f"  {cmd[0]} não encontrado, pulando...")
    
    return True

def create_sample_config():
    """Cria arquivos de configuração de exemplo"""
    import shutil
    
    # Criar .env de exemplo se não existir
    if not os.path.exists('.env') and os.path.exists('.env.example'):
        shutil.copy('.env.example', '.env')
        print(" Arquivo .env criado a partir do .env.example")
    
    # Criar diretório de dados
    data_dir = 'dados_rag'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f" Diretório {data_dir} criado")

if __name__ == '__main__':
    # Se executado diretamente, fazer algumas verificações
    print("Configurando Sistema RAG com Google Gemini...")
    
    # Verificar Python
    print(f" Python {sys.version}")
    
    # Criar configurações de exemplo
    create_sample_config()
    
    print("Setup concluído!")
    print("\nPróximos passos:")
    print("1. pip install -e .")
    print("2. Configure sua GOOGLE_API_KEY no arquivo .env")
    print("3. python rag_gemini_improved.py (ou use os entry points: rag-gemini / rag-server)")
    print("\n Acesse: http://localhost:5000")
