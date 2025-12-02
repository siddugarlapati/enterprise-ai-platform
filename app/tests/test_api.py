import pytest
from httpx import AsyncClient
from fastapi import status
import asyncio

from app.main import app
from app.api.middleware.auth import create_access_token, hash_password

# Test user data
TEST_USER = {
    "username": "testuser",
    "email": "test@example.com",
    "password": "TestPass123!"
}

@pytest.fixture
def anyio_backend():
    return 'asyncio'

@pytest.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.fixture
def auth_headers():
    token = create_access_token({"sub": 1, "email": "test@example.com", "role": "admin"})
    return {"Authorization": f"Bearer {token}"}

# Health Check Tests
@pytest.mark.anyio
async def test_health_check(client):
    response = await client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "version" in data

@pytest.mark.anyio
async def test_root_endpoint(client):
    response = await client.get("/")
    assert response.status_code == status.HTTP_200_OK
    assert "message" in response.json()

@pytest.mark.anyio
async def test_api_info(client):
    response = await client.get("/api/v1/info")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "features" in data
    assert "endpoints" in data

# Auth Tests
@pytest.mark.anyio
async def test_register_user(client):
    response = await client.post("/api/v1/auth/register", json=TEST_USER)
    # May fail if user exists, but should return valid response
    assert response.status_code in [status.HTTP_201_CREATED, status.HTTP_400_BAD_REQUEST]

@pytest.mark.anyio
async def test_login_invalid_credentials(client):
    response = await client.post("/api/v1/auth/login", json={
        "email": "nonexistent@example.com",
        "password": "wrongpassword"
    })
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

@pytest.mark.anyio
async def test_protected_endpoint_without_auth(client):
    response = await client.get("/api/v1/auth/me")
    assert response.status_code == status.HTTP_403_FORBIDDEN

@pytest.mark.anyio
async def test_protected_endpoint_with_auth(client, auth_headers):
    response = await client.get("/api/v1/auth/me", headers=auth_headers)
    # May return 404 if user doesn't exist in test DB
    assert response.status_code in [status.HTTP_200_OK, status.HTTP_404_NOT_FOUND]

# Prediction Tests
@pytest.mark.anyio
async def test_sentiment_analysis(client):
    response = await client.post("/api/v1/predictions/sentiment", json={
        "text": "I love this product! It's amazing."
    })
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "sentiment" in data
    assert "confidence" in data

@pytest.mark.anyio
async def test_ner_extraction(client):
    response = await client.post("/api/v1/predictions/ner", json={
        "text": "Apple Inc. is headquartered in Cupertino, California."
    })
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "entities" in data

@pytest.mark.anyio
async def test_text_classification(client):
    response = await client.post("/api/v1/predictions/classify", json={
        "text": "The stock market crashed today",
        "labels": ["finance", "sports", "technology", "politics"]
    })
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "predicted_label" in data
    assert data["predicted_label"] in ["finance", "sports", "technology", "politics"]

@pytest.mark.anyio
async def test_similarity(client):
    response = await client.post("/api/v1/predictions/similarity", json={
        "text1": "The cat sat on the mat",
        "text2": "A feline rested on the rug"
    })
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "similarity_score" in data
    assert 0 <= data["similarity_score"] <= 1

@pytest.mark.anyio
async def test_embeddings(client):
    response = await client.post("/api/v1/predictions/embeddings", json={
        "texts": ["Hello world", "Goodbye world"]
    })
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "embeddings" in data
    assert len(data["embeddings"]) == 2

# Model Tests
@pytest.mark.anyio
async def test_list_models(client):
    response = await client.get("/api/v1/models/")
    assert response.status_code == status.HTTP_200_OK
    assert isinstance(response.json(), list)

@pytest.mark.anyio
async def test_get_loaded_models(client):
    response = await client.get("/api/v1/models/loaded")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "ml_models" in data
    assert "rag_stats" in data

# Admin Tests (require auth)
@pytest.mark.anyio
async def test_admin_stats(client, auth_headers):
    response = await client.get("/api/v1/admin/stats", headers=auth_headers)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "users" in data
    assert "transactions" in data

@pytest.mark.anyio
async def test_admin_users_list(client, auth_headers):
    response = await client.get("/api/v1/admin/users", headers=auth_headers)
    assert response.status_code == status.HTTP_200_OK

# Rate Limiting Test
@pytest.mark.anyio
async def test_rate_limit_headers(client):
    response = await client.get("/api/v1/info")
    assert "X-RateLimit-Limit" in response.headers
    assert "X-RateLimit-Remaining" in response.headers

# Validation Tests
@pytest.mark.anyio
async def test_sentiment_empty_text(client):
    response = await client.post("/api/v1/predictions/sentiment", json={
        "text": ""
    })
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

@pytest.mark.anyio
async def test_classification_insufficient_labels(client):
    response = await client.post("/api/v1/predictions/classify", json={
        "text": "Some text",
        "labels": ["only_one"]
    })
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
