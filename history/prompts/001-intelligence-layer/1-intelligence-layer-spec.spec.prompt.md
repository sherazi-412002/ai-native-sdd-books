---
id: 1
title: intelligence-layer-spec
stage: spec
date: 2025-12-25
surface: agent
model: Claude Sonnet 4.5
feature: 001-intelligence-layer
branch: 001-intelligence-layer
user: claude
command: sp.specify
labels: ["spec", "intelligence", "rag", "backend"]
links:
  spec: specs/001-intelligence-layer/spec.md
  ticket: null
  adr: null
  pr: null
files:
 - specs/001-intelligence-layer/spec.md
 - specs/001-intelligence-layer/checklists/requirements.md
tests:
 - specs/001-intelligence-layer/checklists/requirements.md
---

## Prompt

```
002-intelligence-layer Implement the FastAPI backend and RAG pipeline with OpenAI ChatKit, Cohere, Qdrant, and Neon.

### 1. Goal
Build a self-hosted RAG backend that serves the Docusaurus chatbot, utilizing Cohere's multilingual embeddings for superior English-Urdu retrieval.

### 2. Infrastructure & Databases
- **FastAPI Server:** Set up in `/server` directory with CORS enabled for the Docusaurus frontend.
- **Qdrant Cloud:** Initialize a collection named `book_content` with 1024 dimensions (optimized for Cohere `embed-multilingual-v3.0`) and Cosine similarity.
- **Neon Postgres:** Define a schema for `chat_sessions` and `messages` to persist history beyond the frontend state.

### 3. The RAG Pipeline (The "Brain")
- **Embedding Engine:** Integrate Cohere SDK. Use `model="embed-multilingual-v3.0"`.
  - Use `input_type="search_document"` for ingestion.
  - Use `input_type="search_query"` for user questions.
- **Orchestration:** Use the OpenAI Agents SDK to create a "Book Expert" agent.
- **Context Awareness (Selected Text):**
  - The `/chat` endpoint must accept an optional `selected_text` field.
  - Logic: If `selected_text` is present, it must be injected as a "High Priority Context" block in the system prompt to ground the agent's response.

### 4. Data Ingestion Script
- Create a Python script `scripts/ingest_docs.py` that:
  - Recursively parses `.md` and `.mdx` files from the Phase 1 `docs/` folder.
  - Chunks content by sub-chapters/headers.
  - Generates embeddings via Cohere and upserts them to Qdrant with metadata (title, module, chapter_url).

### 5. Frontend-Backend Integration
- Connect the `ChatWidget` component from Phase 1 to the FastAPI `/chat` endpoint.
- Implement streaming responses using the OpenAI ChatKit SDK's streaming capabilities.
- Add a "Source Citation" UI element in the chat window to display the `chapter_url` from Qdrant metadata.

### 6. Technical Constraints
- Use `pydantic` for strict request/response validation.
- Store all API keys (OpenAI, Cohere, Qdrant, Neon) in a `.env` file (ensure it is git-ignored).
- Implement a simple "Ready" check endpoint for the chatbot to verify database connectivity.
```

## Response snapshot

Created a comprehensive specification for the intelligence layer feature, implementing a FastAPI backend and RAG pipeline. The specification includes:

1. User scenarios for core chat functionality with context awareness
2. Data ingestion and indexing workflows
3. Session and message persistence requirements
4. Functional requirements covering all technical aspects (FastAPI, Cohere embeddings, Qdrant, OpenAI agents, etc.)
5. Key entities (ChatSession, Message, DocumentChunk, BookContent)
6. Success criteria with measurable outcomes

The specification was validated against quality criteria and all requirements were met.

## Outcome

- ✅ Impact: Created complete feature specification for RAG backend system
- 🧪 Tests: Validation checklist confirms spec completeness
- 📁 Files: Created spec.md and requirements checklist
- 🔁 Next prompts: Ready for clarification or planning phase
- 🧠 Reflection: Spec successfully abstracts technical implementation while preserving user value

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): PASS
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A