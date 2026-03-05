# AZURE_RAG_APP

Agentic RAG system with Azure Blob Storage and Event Grid integration.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure environment variables in `.env`.

## Testing

Run tests locally:
```bash
PYTHONPATH=. pytest tests/
```

## Docker

Build the image:
```bash
docker build -t azure-rag-app .
```

Run the container:
```bash
docker run -d \
  --name rag-consumer \
  --env-file .env \
  azure-rag-app
```

## CI/CD

This repository uses GitHub Actions for automated testing on every push to the `main` branch.
