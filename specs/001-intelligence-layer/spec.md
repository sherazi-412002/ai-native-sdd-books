# Feature Specification: Intelligence Layer - RAG Backend for Docusaurus Chatbot

**Feature Branch**: `001-intelligence-layer`
**Created**: 2025-12-25
**Status**: Draft
**Input**: User description: "Implement the FastAPI backend and RAG pipeline with OpenAI ChatKit, Cohere, Qdrant, and Neon. Build a self-hosted RAG backend that serves the Docusaurus chatbot, utilizing Cohere's multilingual embeddings for superior English-Urdu retrieval."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Chat with Book Expert Agent (Priority: P1)

A user interacts with the Docusaurus chatbot to ask questions about book content. The user can optionally select text in the documentation that gets injected as high-priority context for more targeted responses. The chatbot provides accurate answers with source citations.

**Why this priority**: This is the core value proposition - enabling users to get answers from book content through natural language queries.

**Independent Test**: Can be fully tested by asking questions to the chatbot and verifying that responses are accurate and include source citations.

**Acceptance Scenarios**:

1. **Given** user has access to the Docusaurus chatbot, **When** user submits a question about book content, **Then** the system returns an accurate response with source citations
2. **Given** user has selected text in the documentation, **When** user submits a question with selected text context, **Then** the system prioritizes the selected text context in its response

---

### User Story 2 - Data Ingestion and Indexing (Priority: P1)

A system administrator runs the data ingestion script to index book content into the RAG system. The system processes markdown and MDX files, chunks content by sections, generates embeddings, and stores them in the vector database.

**Why this priority**: Without proper data ingestion, the chatbot cannot provide accurate responses to user queries.

**Independent Test**: Can be fully tested by running the ingestion script and verifying that content is properly indexed in the vector database.

**Acceptance Scenarios**:

1. **Given** markdown/MDX documentation files exist, **When** ingestion script is executed, **Then** content is properly chunked and stored in the vector database
2. **Given** documentation content exists, **When** embeddings are generated, **Then** they are stored with proper metadata (title, module, chapter_url)

---

### User Story 3 - Session and Message Persistence (Priority: P2)

A user continues a conversation with the chatbot across multiple sessions. The system maintains chat history beyond the frontend state by persisting conversations in a database.

**Why this priority**: Enhances user experience by allowing persistent conversation history across sessions.

**Independent Test**: Can be fully tested by creating a conversation, ending the session, and resuming to verify history persistence.

**Acceptance Scenarios**:

1. **Given** user starts a new chat session, **When** conversation occurs, **Then** messages are persisted in the database
2. **Given** user returns to a previous session, **When** session is loaded, **Then** previous conversation history is available

---

### Edge Cases

- What happens when the vector database is unavailable during chat requests?
- How does the system handle malformed markdown files during ingestion?
- What occurs when API keys are invalid or rate limits are exceeded?
- How does the system handle very large documents during chunking?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a FastAPI backend server with CORS enabled for Docusaurus frontend integration
- **FR-002**: System MUST integrate Cohere SDK to generate embeddings using model="embed-multilingual-v3.0" with input_type="search_document" for ingestion and input_type="search_query" for user questions
- **FR-003**: System MUST create and maintain a Qdrant collection named "book_content" with 1024 dimensions and Cosine similarity
- **FR-004**: System MUST implement an OpenAI Agents SDK "Book Expert" agent for RAG orchestration
- **FR-005**: System MUST provide a /chat endpoint that accepts an optional "selected_text" field and injects it as high-priority context in the system prompt
- **FR-006**: System MUST include a data ingestion script that recursively parses .md and .mdx files from docs/ folder, chunks content by sub-chapters/headers, and upserts embeddings to Qdrant with metadata
- **FR-007**: System MUST stream responses to the frontend using OpenAI ChatKit SDK's streaming capabilities
- **FR-008**: System MUST display source citations in the chat UI showing the chapter_url from Qdrant metadata
- **FR-009**: System MUST define and maintain Neon Postgres schema for chat_sessions and messages tables
- **FR-010**: System MUST implement a "Ready" health check endpoint to verify database connectivity
- **FR-011**: System MUST use pydantic for strict request/response validation
- **FR-012**: System MUST securely store API keys (OpenAI, Cohere, Qdrant, Neon) in a .env file that is git-ignored

### Key Entities

- **ChatSession**: Represents a user's conversation session, containing metadata and relationships to associated messages
- **Message**: Represents an individual message in a conversation, including sender type (user/agent), content, timestamp, and session reference
- **DocumentChunk**: Represents a segment of book content that has been processed and embedded, with associated metadata (title, module, chapter_url) for retrieval
- **BookContent**: Represents the indexed book content in the Qdrant vector database with multilingual embeddings

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can receive accurate answers to book-related questions through the Docusaurus chatbot with response times under 3 seconds
- **SC-002**: System successfully indexes 100% of markdown and MDX documentation files with proper chunking and metadata preservation
- **SC-003**: 95% of user queries return relevant responses with accurate source citations
- **SC-004**: System maintains conversation history persistence with 99.9% data integrity
- **SC-005**: The RAG pipeline demonstrates superior English-Urdu retrieval performance compared to standard embeddings