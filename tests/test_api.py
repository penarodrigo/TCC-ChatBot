import pytest
from rag_gemini_system.app import create_app

@pytest.fixture
def app():
    """Create and configure a new app instance for each test."""
    app = create_app()
    app.config.update({
        "TESTING": True,
    })
    yield app

@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()

def test_health_check_endpoint(client):
    """Testa o endpoint de health check '/health'."""
    response = client.get('/health')
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "Online!"
    assert "message" in json_data

def test_ask_endpoint_missing_question(client):
    """Testa o endpoint /api/ask sem o campo 'question'."""
    response = client.get('/api/ask')
    assert response.status_code == 400
    json_data = response.get_json()
    assert "'question' como query parameter é obrigatório" in json_data["error"]

def test_ask_endpoint_empty_question(client):
    """Testa o endpoint /api/ask com uma 'question' vazia."""
    response = client.get('/api/ask?question=%20%20')
    assert response.status_code == 400
    json_data = response.get_json()
    assert "'question' como query parameter é obrigatório." in json_data["error"]

# This test needs to be updated to mock the RAG system
# as it will try to load the models, which is slow and requires an API key.
@pytest.mark.skip(reason="Needs mocking of the RAG system to avoid loading models.")
def test_ask_endpoint_valid_question(client):
    """Testa o endpoint /api/ask com uma pergunta válida (resposta placeholder)."""
    test_question = "Qual o sentido da vida?"
    # Mock the get_rag_system function here to return a mock RAG system
    response = client.get(f'/api/ask?question={test_question}')
    assert response.status_code == 200
    # Streaming response is not easily testable here, so we just check the status code