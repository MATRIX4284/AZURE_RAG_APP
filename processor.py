import os
import tempfile
import logging
from urllib.parse import unquote
from pathlib import Path
from dotenv import load_dotenv

# Docling imports
from docling.document_converter import DocumentConverter

# LangChain imports
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Azure Storage imports
from azure.storage.blob import BlobServiceClient

load_dotenv()

# Configuration
VECTOR_STORE_PATH = "faiss_index"

# Configure logging to a file
logging.basicConfig(
    filename='processing_failures.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_embeddings_client():
    """
    Initializes the Azure OpenAI Embeddings client with the correct token scope for AI Foundry.
    """
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    
    # For AI Foundry Projects, we need the Machine Learning scope
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://ai.azure.com/.default"
    )
    
    return AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-large"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_ad_token_provider=token_provider,
    )

def log_failure(blob_url, error_message):
    """Logs the failed blob URL and error to the log file."""
    logging.error(f"Failed to process: {blob_url} | Error: {error_message}")
    print(f"FAILURE LOGGED: {blob_url}")

def process_pdf_to_faiss(blob_url: str):
    """
    End-to-end pipeline: Download -> Parse -> Chunk -> Index.
    """
    try:
        print(f"Starting processing for: {blob_url}")
        
        # 1. Download the file from Blob Storage
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if not connection_string:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING is missing")
            
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        # Robust URL parsing to handle folders and special characters
        # URL format: https://<account>.blob.core.windows.net/<container>/<path/to/blob>
        from urllib.parse import urlparse
        parsed_url = urlparse(blob_url)
        path_parts = parsed_url.path.lstrip('/').split('/')
        
        if len(path_parts) < 2:
            raise ValueError(f"Could not parse container and blob from URL: {blob_url}")
            
        container_name = unquote(path_parts[0])
        blob_name = unquote('/'.join(path_parts[1:]))
        
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir) / Path(blob_name).name
            
            print(f"Downloading from Container: '{container_name}', Blob: '{blob_name}'")
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
            blob_data = blob_client.download_blob().readall()
            
            # Check if we downloaded an Azure Error XML instead of a PDF
            if b"<?xml" in blob_data and b"<Error>" in blob_data:
                error_text = blob_data.decode('utf-8', errors='ignore')
                raise ValueError(f"Azure Storage Error (check permissions/URL): {error_text}")

            with open(local_path, "wb") as f:
                f.write(blob_data)
                
            # Diagnostic checks
            file_size = os.path.getsize(local_path)
            print(f"Downloaded file size: {file_size} bytes")
            
            if file_size == 0:
                raise ValueError("Downloaded file is empty")

            with open(local_path, "rb") as f:
                header = f.read(4)
                print(f"File header bytes: {header}")
                if header != b"%PDF":
                    print("WARNING: File does not start with %PDF. It might not be a valid PDF.")

            # 2. Parse with Docling
            print("Parsing PDF with Docling (converting to Markdown)...")
            print("PDF_PATH")
            print(str(local_path))
            converter = DocumentConverter()
            result = converter.convert(str(local_path))
            markdown_content = result.document.export_to_markdown()

            # 3. Chunk with LangChain (Markdown Header Splitting)
            print("Chunking document with MarkdownHeaderTextSplitter...")
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            chunks = markdown_splitter.split_text(markdown_content)
            
            # Add metadata (source URL) to each chunk
            for chunk in chunks:
                chunk.metadata["source"] = blob_url

            print(f"Created {len(chunks)} chunks.")

            # 4. Initialize Embeddings and Vector Store
            embeddings = get_embeddings_client()
            
            print("Updating FAISS index...")
            if os.path.exists(VECTOR_STORE_PATH):
                # Load existing index
                vectorstore = FAISS.load_local(
                    VECTOR_STORE_PATH, 
                    embeddings, 
                    allow_dangerous_deserialization=True
                )
                # Add new chunks
                vectorstore.add_documents(chunks)
            else:
                # Create new index
                vectorstore = FAISS.from_documents(chunks, embeddings)
            
            # Save index locally
            vectorstore.save_local(VECTOR_STORE_PATH)
            print(f"Success! FAISS index updated and saved to '{VECTOR_STORE_PATH}'.")

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"\nCRITICAL ERROR: {error_msg}")
        log_failure(blob_url, error_msg)

if __name__ == "__main__":
    # Test script for local debugging if needed
    # test_url = "https://youraccount.blob.core.windows.net/uploads/test.pdf"
    # process_pdf_to_faiss(test_url)
    pass
