import pytest
import asyncio
import json
from unittest.mock import MagicMock, patch
from consume_blob_events import process_message

@pytest.mark.asyncio
async def test_process_message_pdf(mock_env_vars, mocker):
    # Setup mocks
    mock_queue_client = MagicMock()
    
    # Mock message content
    event_data = {
        "subject": "/blobServices/default/containers/test/blobs/doc.pdf",
        "eventType": "Microsoft.Storage.BlobCreated",
        "data": {"url": "https://test.blob.core.windows.net/test/doc.pdf"}
    }
    mock_msg = MagicMock()
    mock_msg.content = json.dumps([event_data])
    
    # Mock process_pdf_to_faiss in the module
    mock_process = mocker.patch("consume_blob_events.process_pdf_to_faiss")
    
    # Run
    await process_message(mock_queue_client, mock_msg)
    
    # Assert
    mock_process.assert_called_once_with("https://test.blob.core.windows.net/test/doc.pdf")
    mock_queue_client.delete_message.assert_called_once_with(mock_msg)

@pytest.mark.asyncio
async def test_process_message_non_pdf(mock_env_vars, mocker):
    # Setup mocks
    mock_queue_client = MagicMock()
    
    # Mock message content
    event_data = {
        "subject": "/blobServices/default/containers/test/blobs/doc.txt",
        "eventType": "Microsoft.Storage.BlobCreated",
        "data": {"url": "https://test.blob.core.windows.net/test/doc.txt"}
    }
    mock_msg = MagicMock()
    mock_msg.content = json.dumps(event_data)
    
    # Mock process_pdf_to_faiss
    mock_process = mocker.patch("consume_blob_events.process_pdf_to_faiss")
    
    # Run
    await process_message(mock_queue_client, mock_msg)
    
    # Assert
    mock_process.assert_not_called()
    mock_queue_client.delete_message.assert_called_once_with(mock_msg)
