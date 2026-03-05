import pytest
import os
from unittest.mock import MagicMock

@pytest.fixture
def mock_env_vars(monkeypatch):
    monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=test;EndpointSuffix=core.windows.net")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-large")

@pytest.fixture
def mock_blob_service_client(mocker):
    return mocker.patch("processor.BlobServiceClient")

@pytest.fixture
def mock_embeddings(mocker):
    return mocker.patch("processor.AzureOpenAIEmbeddings")

@pytest.fixture
def mock_faiss(mocker):
    return mocker.patch("processor.FAISS")

@pytest.fixture
def mock_document_converter(mocker):
    return mocker.patch("processor.DocumentConverter")
