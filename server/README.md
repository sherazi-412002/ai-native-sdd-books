# Intelligence Layer - RAG Backend

This is the backend service for the Intelligence Layer feature, providing a RAG (Retrieval-Augmented Generation) system for the Docusaurus chatbot. The system uses Cohere's multilingual embeddings for English-Urdu retrieval, OpenAI Agents for orchestration, Qdrant for vector storage, and Neon Postgres for conversation history.

## Features

- **RAG Pipeline**: Uses Cohere embeddings and OpenAI agents to provide accurate answers based on book content
- **Multilingual Support**: Optimized for English-Urdu retrieval using Cohere's `embed-multilingual-v3.0` model
- **Session Management**: Persistent chat sessions with message history
- **Source Citations**: Responses include citations with URLs to original sources
- **Context Awareness**: Supports selected text context injection for more targeted responses
- **Streaming Responses**: Real-time response streaming to the frontend
- **Data Ingestion**: Script to process documentation files and index them for retrieval

## Architecture

The system follows a service-oriented architecture:

- **API Layer**: FastAPI endpoints for chat, ingestion, and health checks
- **Service Layer**: Business logic for RAG orchestration, embedding generation, and database operations
- **Data Layer**: Neon Postgres for session management, Qdrant for vector storage
- **Utility Layer**: Text processing, validation, and logging utilities

## Setup

### Prerequisites

- Python 3.11+
- pip package manager
- Access to the following APIs:
  - OpenAI API key
  - Cohere API key
  - Qdrant Cloud account
  - Neon Postgres account

### Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure environment variables by copying `.env.example` to `.env` and filling in your API keys

### Environment Variables

Create a `.env` file with the following variables:

```env
OPENAI_API_KEY=your_openai_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_URL=your_qdrant_cloud_url_here
NEON_DATABASE_URL=your_neon_postgres_connection_string_here
DEBUG=False
LOG_LEVEL=INFO
```

## Usage

### Running the Server

```bash
cd server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

### API Endpoints

- `POST /api/v1/chat` - Chat with the Book Expert agent
- `POST /api/v1/chat/stream` - Streaming chat endpoint
- `GET /api/v1/chat/sessions/{session_id}` - Get chat session details
- `GET /api/v1/health/ready` - Health check endpoint
- `GET /api/v1/health/live` - Liveness check endpoint

### Data Ingestion

To ingest documentation content:

```bash
python scripts/ingest_docs.py --source-path docs --recursive --file-types .md .mdx
```

## Configuration

### Server Configuration

- Port: Default is 8000
- CORS settings: Configured for Docusaurus frontend integration
- API timeout: Configured for optimal response times
- Rate limiting: Configured based on API provider limits

### Embedding Configuration

- Model: `embed-multilingual-v3.0` (Cohere)
- Input type: `search_document` for ingestion, `search_query` for questions
- Dimensions: 1024 (for Qdrant compatibility)

## Development

### Running Tests

```bash
pytest tests/
```

### API Documentation

The API documentation is automatically available at:
- `http://localhost:8000/docs` - Interactive API documentation (Swagger UI)
- `http://localhost:8000/redoc` - Alternative API documentation (ReDoc)

### Database Migrations

To run database migrations:

```bash
cd server
alembic upgrade head
```

To create a new migration:

```bash
alembic revision --autogenerate -m "Description of changes"
```

## Security

- API keys are loaded from environment variables and not committed to the repository
- Input validation is performed on all user inputs
- SQL injection protection through SQLAlchemy ORM
- Rate limiting to prevent abuse of API endpoints