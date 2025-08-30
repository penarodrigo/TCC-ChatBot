import pytest
from rag_gemini_system.app import create_app, conversation_history

@pytest.fixture
def app():
    """Create and configure a new app instance for each test."""
    app = create_app()
    app.config.update({
        "TESTING": True,
    })
    # Clear history before each test
    conversation_history.clear()
    yield app

@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()

def test_get_history_empty(client):
    """Test that the /api/history endpoint returns an empty list when no conversations have occurred."""
    response = client.get('/api/history')
    assert response.status_code == 200
    json_data = response.get_json()
    assert "history" in json_data
    assert isinstance(json_data["history"], list)
    assert len(json_data["history"]) == 0

# This test will require mocking the RAG system to avoid actual model loading and API calls
# For now, we'll simulate adding to history directly or by making a simple call to /api/ask
# that doesn't rely on the RAG system's full functionality.
def test_get_history_with_conversations(client):
    """Test that the /api/history endpoint returns the correct conversation history after some interactions."""
    # Simulate a conversation turn by directly adding to the history (for testing purposes)
    # In a real scenario, this would be populated by calls to /api/ask
    conversation_history.append({
        "timestamp": "2025-08-26T10:00:00.000000",
        "question": "Hello",
        "answer": "Hi there!"
    })
    conversation_history.append({
        "timestamp": "2025-08-26T10:01:00.000000",
        "question": "How are you?",
        "answer": "I'm doing great!"
    })

    response = client.get('/api/history')
    assert response.status_code == 200
    json_data = response.get_json()
    assert "history" in json_data
    assert isinstance(json_data["history"], list)
    assert len(json_data["history"]) == 2

    assert json_data["history"][0]["question"] == "Hello"
    assert json_data["history"][0]["answer"] == "Hi there!"
    assert json_data["history"][1]["question"] == "How are you?"
    assert json_data["history"][1]["answer"] == "I'm doing great!"
