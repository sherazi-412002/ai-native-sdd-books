# Quickstart Guide: Intelligence Layer - RAG Backend

This guide will help you get the Intelligence Layer backend up and running quickly.

## Prerequisites

- Python 3.11 or higher
- pip package manager
- Git
- Access to the following APIs:
  - OpenAI API key
  - Cohere API key
  - Qdrant Cloud account
  - Neon Postgres account

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Set up Python Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root with your API keys:

```env
OPENAI_API_KEY=your_openai_api_key
COHERE_API_KEY=your_cohere_api_key
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_URL=your_qdrant_cloud_url
NEON_DATABASE_URL=your_neon_postgres_connection_string
```

## Running the Application

### 1. Start the Server

```bash
cd server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

### 2. Verify Installation

Check the health endpoint:
```bash
curl http://localhost:8000/api/v1/health/ready
```

## Ingesting Documentation

### 1. Prepare Documentation Files

Place your `.md` and `.mdx` documentation files in a `docs/` directory.

### 2. Run the Ingestion Script

```bash
python scripts/ingest_docs.py --source-path docs --recursive --file-types .md .mdx
```

This will:
- Parse all `.md` and `.mdx` files
- Chunk content by sub-chapters/headers
- Generate embeddings using Cohere
- Upsert to Qdrant with metadata

## Using the Chat API

### Send a Message

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the main concept in chapter 1?",
    "session_id": "123e4567-e89b-12d3-a456-426614174000",
    "selected_text": "Optional context from selected text"
  }'
```

### Streaming Response

```bash
curl -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the main concept in chapter 1?",
    "session_id": "123e4567-e89b-12d3-a456-426614174000"
  }'
```

## API Documentation

View the interactive API documentation at:
- `http://localhost:8000/docs` (Swagger UI)
- `http://localhost:8000/redoc` (ReDoc)

## Database Migrations

Run database migrations:
```bash
cd server
alembic upgrade head
```

## Testing the Installation

1. Verify the health check returns "ready":
   ```bash
   curl http://localhost:8000/api/v1/health/ready
   ```

2. Test a simple chat request:
   ```bash
   curl -X POST http://localhost:8000/api/v1/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "Hello"}'
   ```

3. Check that the response includes sources (after ingesting documentation).

## Troubleshooting

### Common Issues

- **API Keys**: Ensure all required API keys are set in the `.env` file
- **Database Connection**: Verify the Neon Postgres connection string is correct
- **Qdrant Connection**: Ensure the Qdrant URL and API key are correct
- **Port Conflicts**: If port 8000 is in use, change the port in the uvicorn command

### Logs

Check the logs in the `logs/` directory for detailed error information.