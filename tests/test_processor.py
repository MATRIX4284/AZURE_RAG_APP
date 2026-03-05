import pytest
from processor import process_pdf_to_faiss
from unittest.mock import patch, MagicMock

def test_process_pdf_to_faiss_basic(mock_env_vars, mock_blob_service_client, mock_document_converter, mock_embeddings, mock_faiss):
    # Setup mocks
    mock_blob_client = MagicMock()
    mock_blob_service_client.from_connection_string.return_value.get_blob_client.return_value = mock_blob_client
    mock_blob_client.download_blob.return_value.readall.return_value = b"%PDF-1.4\n%..."
    
    mock_doc = MagicMock()
    mock_doc.export_to_markdown.return_value = "# Test Header\nTest content"
    mock_document_converter.return_value.convert.return_value.document = mock_doc
    
    # Mock os.path.exists to return False to trigger from_documents
    with patch("processor.os.path.exists", return_value=False):
        # Run
        process_pdf_to_faiss("https://test.blob.core.windows.net/container/test.pdf")
    
    # Assert
    mock_blob_service_client.from_connection_string.assert_called_once()
    mock_document_converter.return_value.convert.assert_called_once()
    mock_faiss.from_documents.assert_called()
