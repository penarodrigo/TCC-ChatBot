# c:\Users\c057545\Downloads\CBCODIMPROVED\tests\test_api.py
import pytest
from rag_gemini_improved import app as flask_app # Importe sua instância Flask

@pytest.fixture
def client():
    """Cria um cliente de teste para a aplicação Flask."""
    flask_app.config['TESTING'] = True
    # Outras configurações específicas para teste podem ser adicionadas aqui
    # Ex: flask_app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    
    with flask_app.test_client() as client:
        yield client

def test_health_check_endpoint(client):
    """Testa o endpoint de health check '/'."""
    response = client.get('/')
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "Sistema RAG com Gemini está em execução!"
    assert "message" in json_data

def test_ask_endpoint_missing_question(client):
    """Testa o endpoint /api/ask sem o campo 'question'."""
    response = client.post('/api/ask', json={})
    assert response.status_code == 400
    json_data = response.get_json()
    assert "O campo 'question' é obrigatório" in json_data["error"]

def test_ask_endpoint_empty_question(client):
    """Testa o endpoint /api/ask com uma 'question' vazia."""
    response = client.post('/api/ask', json={"question": "  "})
    assert response.status_code == 400
    json_data = response.get_json()
    assert "'question' deve ser uma string não vazia" in json_data["error"]

def test_ask_endpoint_valid_question(client):
    """Testa o endpoint /api/ask com uma pergunta válida (resposta placeholder)."""
    test_question = "Qual o sentido da vida?"
    response = client.post('/api/ask', json={"question": test_question})
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["question"] == test_question
    assert "Esta é uma resposta placeholder" in json_data["answer"]
    # Quando você implementar a lógica real do RAG, ajuste este assert
    # para verificar a resposta esperada.
