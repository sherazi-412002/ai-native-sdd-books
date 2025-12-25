# Quickstart Guide: Intelligence Layer - RAG Backend for Docusaurus Chatbot

## Prerequisites

- Python 3.11 or higher
- pip package manager
- Git
- Access to the following APIs:
  - OpenAI API key
  - Cohere API key
  - Qdrant Cloud account
  - Neon Postgres account

## Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root with the following variables:

```env
OPENAI_API_KEY=your_openai_api_key
COHERE_API_KEY=your_cohere_api_key
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key
NEON_DATABASE_URL=your_neon_postgres_connection_string
```

## Running the Application

### 1. Start the FastAPI Server

```bash
cd server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

### 2. Run Database Migrations (if applicable)

```bash
# Run database migrations
python -m alembic upgrade head
```

## Ingest Documentation Content

### 1. Prepare Documentation Files

Ensure your documentation files (`.md` and `.mdx`) are in the `docs/` directory.

### 2. Run the Ingestion Script

```bash
python scripts/ingest_docs.py --source-path docs/ --recursive
```

This will:
- Parse all `.md` and `.mdx` files
- Chunk content by sub-chapters/headers
- Generate embeddings using Cohere
- Upsert to Qdrant with metadata

## Using the API

### Chat Endpoint

Send a message to the chatbot:

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the main concept in chapter 1?",
    "session_id": "123e4567-e89b-12d3-a456-426614174000",
    "selected_text": "Optional context from selected text"
  }'
```

### Health Check

Verify the service is ready:

```bash
curl http://localhost:8000/api/v1/health/ready
```

## Development

### Running Tests

```bash
pytest tests/
```

### API Documentation

The API documentation is automatically available at:
- `http://localhost:8000/docs` - Interactive API documentation (Swagger UI)
- `http://localhost:8000/redoc` - Alternative API documentation (ReDoc)

## Configuration

### Server Configuration

The server configuration can be modified in `server/config/settings.py`:

- Port: Default is 8000
- CORS settings: Configured for Docusaurus frontend integration
- API timeout: Configured for optimal response times
- Rate limiting: Configured based on API provider limits

### Embedding Configuration

The embedding settings are in `server/services/embedding.py`:

- Model: `embed-multilingual-v3.0` (Cohere)
- Input type: `search_document` for ingestion, `search_query` for questions
- Dimensions: 1024 (for Qdrant compatibility)