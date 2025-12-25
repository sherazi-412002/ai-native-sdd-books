# Research: Intelligence Layer - RAG Backend for Docusaurus Chatbot

## Decision: FastAPI Framework Choice
**Rationale**: FastAPI was chosen as the web framework based on the feature specification requirements. It provides excellent support for async operations, built-in OpenAPI documentation, and high performance which are essential for a RAG backend handling concurrent requests and streaming responses.

**Alternatives considered**:
- Flask: Less performant for async operations
- Django: Overkill for API-only backend
- Express.js: Would require switching to Node.js ecosystem

## Decision: Cohere Embed-multilingual-v3.0 Model
**Rationale**: The specification explicitly requires Cohere's `embed-multilingual-v3.0` model optimized for English-Urdu retrieval. This model has 1024 dimensions which matches the Qdrant collection requirements specified.

**Alternatives considered**:
- OpenAI embeddings: Don't support multilingual as well as Cohere
- Sentence Transformers: Self-hosting complexity
- Other Cohere models: v3.0 specifically mentioned in requirements

## Decision: Qdrant Vector Database
**Rationale**: Qdrant was specified in the requirements with 1024 dimensions and Cosine similarity. It's a cloud-native vector database that supports metadata storage for source citations, making it ideal for RAG applications.

**Alternatives considered**:
- Pinecone: Vendor lock-in concerns
- Weaviate: More complex setup
- PostgreSQL with pgvector: Less optimized for vector operations

## Decision: Neon Postgres for Session Storage
**Rationale**: Neon was specified in the requirements for storing chat_sessions and messages. It's a serverless Postgres that scales automatically and integrates well with Python applications.

**Alternatives considered**:
- MongoDB: Would add complexity with different data model
- Redis: Not ideal for structured session data
- SQLite: Not suitable for production web application

## Decision: OpenAI Agents SDK for Orchestration
**Rationale**: The specification requires using OpenAI Agents SDK to create a "Book Expert" agent for RAG orchestration. This provides the AI reasoning capabilities needed for the RAG pipeline.

**Alternatives considered**:
- LangChain: Would require different architecture
- Custom implementation: More complex than using OpenAI's solution
- Anthropic agents: Not specified in requirements

## Decision: Data Ingestion Approach
**Rationale**: The ingestion script will recursively parse .md and .mdx files, chunk by headers/sub-chapters, and store with metadata. This approach preserves document structure while enabling effective retrieval.

**Alternatives considered**:
- Chunk by fixed token length: Might break semantic boundaries
- Process as whole documents: Less precise retrieval
- Real-time parsing: Would be too slow for large documents

## Decision: Streaming Implementation
**Rationale**: Using OpenAI ChatKit SDK's streaming capabilities will provide real-time responses to the frontend, improving user experience with large responses.

**Alternatives considered**:
- Server-sent events: Less direct integration with OpenAI tools
- WebSockets: More complex implementation
- Chunked responses: Less efficient than native streaming