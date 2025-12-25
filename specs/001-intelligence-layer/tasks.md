---
description: "Task list for Intelligence Layer - RAG Backend for Docusaurus Chatbot implementation"
---

# Tasks: Intelligence Layer - RAG Backend for Docusaurus Chatbot

**Input**: Design documents from `/specs/001-intelligence-layer/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: The examples below include test tasks. Tests are OPTIONAL - only include them if explicitly requested in the feature specification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Web backend**: `server/`, `scripts/` at repository root
- **Dependencies**: `requirements.txt` at root
- **Environment**: `.env` at root
- Paths adjusted based on plan.md structure

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [x] T001 Create project structure per implementation plan in server/ directory
- [x] T002 Initialize Python 3.11 project with FastAPI, Cohere SDK, OpenAI SDK, Qdrant, SQLAlchemy, Pydantic dependencies in requirements.txt
- [ ] T003 [P] Configure linting and formatting tools (black, flake8, mypy)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T004 Setup configuration management with settings.py in server/config/
- [x] T005 [P] Setup database connection framework for Neon Postgres in server/config/database.py
- [x] T006 [P] Setup Qdrant connection framework for vector storage in server/services/vector_store.py
- [x] T007 Create base models for ChatSession and Message entities in server/models/chat.py
- [x] T008 Create base models for DocumentChunk entity in server/models/document.py
- [x] T009 Create Pydantic schemas for API validation in server/models/schemas.py
- [x] T010 Configure CORS middleware for Docusaurus frontend integration in server/main.py
- [x] T011 Setup environment configuration management with .env file
- [x] T012 Create embedding service using Cohere SDK in server/services/embedding.py
- [x] T013 Setup error handling and logging infrastructure

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Chat with Book Expert Agent (Priority: P1) 🎯 MVP

**Goal**: Enable users to interact with the Docusaurus chatbot to ask questions about book content with optional selected text context and receive responses with source citations

**Independent Test**: Can be fully tested by asking questions to the chatbot and verifying that responses are accurate and include source citations.

### Implementation for User Story 1

- [x] T014 [P] [US1] Implement RAG orchestration service in server/services/rag.py
- [x] T015 [P] [US1] Create OpenAI Agents SDK "Book Expert" agent implementation in server/services/rag.py
- [x] T016 [US1] Implement /api/v1/chat endpoint in server/api/v1/chat.py
- [x] T017 [US1] Add selected text context injection logic in system prompt for /chat endpoint
- [x] T018 [US1] Implement streaming responses using OpenAI ChatKit SDK in /chat endpoint
- [x] T019 [US1] Add source citation display functionality to include chapter_url from Qdrant metadata
- [x] T020 [US1] Add request/response validation for chat endpoint using Pydantic schemas
- [x] T021 [US1] Add logging for chat operations
- [x] T022 [US1] Implement basic session management for chat continuity

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Data Ingestion and Indexing (Priority: P1)

**Goal**: Enable system administrators to run data ingestion script to index book content into the RAG system with proper chunking and metadata storage

**Independent Test**: Can be fully tested by running the ingestion script and verifying that content is properly indexed in the vector database.

### Implementation for User Story 2

- [x] T023 [P] [US2] Create text chunking utilities by headers/sub-chapters in server/utils/text_splitter.py
- [x] T024 [P] [US2] Create input validation utilities in server/utils/validators.py
- [x] T025 [US2] Implement data ingestion script that recursively parses .md and .mdx files in scripts/ingest_docs.py
- [x] T026 [US2] Add Cohere embedding generation for ingestion with input_type="search_document" in ingestion script
- [x] T027 [US2] Implement upsert to Qdrant with metadata (title, module, chapter_url) in ingestion script
- [x] T028 [US2] Add error handling for malformed markdown files during ingestion
- [x] T029 [US2] Add logging for ingestion operations
- [x] T030 [US2] Add progress tracking and status reporting to ingestion script

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Session and Message Persistence (Priority: P2)

**Goal**: Enable users to continue conversations across multiple sessions by persisting chat history beyond frontend state

**Independent Test**: Can be fully tested by creating a conversation, ending the session, and resuming to verify history persistence.

### Implementation for User Story 3

- [x] T031 [P] [US3] Implement database operations service for chat persistence in server/services/database.py
- [x] T032 [US3] Add Neon Postgres schema for chat_sessions and messages tables
- [x] T033 [US3] Implement message persistence logic in chat endpoint
- [x] T034 [US3] Add session creation and retrieval functionality
- [x] T035 [US3] Implement session history loading in chat endpoint
- [x] T036 [US3] Add database migration framework if needed
- [x] T037 [US3] Add data integrity checks for chat persistence
- [x] T038 [US3] Add logging for session operations

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Health Check and Ready Endpoint

**Goal**: Implement a "Ready" health check endpoint to verify database connectivity as specified in requirements

**Independent Test**: Can be tested by calling the health endpoint and verifying it returns proper status when dependencies are available.

### Implementation for Health Check

- [x] T039 [P] [HC] Implement health check endpoint in server/api/v1/health.py
- [x] T040 [HC] Add database connectivity verification to health endpoint
- [x] T041 [HC] Add Qdrant connectivity verification to health endpoint
- [x] T042 [HC] Add API key validation to health endpoint
- [x] T043 [HC] Add comprehensive service status reporting

**Checkpoint**: Health check functionality should be working independently

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [x] T044 [P] Documentation updates in docs/ and server/README.md
- [ ] T045 Code cleanup and refactoring
- [ ] T046 Performance optimization across all stories
- [x] T047 Security hardening for API key management
- [x] T048 Run quickstart.md validation
- [x] T049 Add comprehensive error handling and graceful degradation
- [x] T050 Add rate limiting for API endpoints
- [x] T051 Add monitoring and metrics collection

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 3 (P2)**: Can start after Foundational (Phase 2) - May integrate with US1 but should be independently testable
- **Health Check (Phase 6)**: Can start after Foundational (Phase 2) - No dependencies on other stories

### Within Each User Story

- Core implementation before integration
- Story complete before moving to next priority
- Each story should be independently testable

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- Different user stories can be worked on in parallel by different team members

---

## Parallel Example: User Story 1

```bash
# Launch all parallel tasks for User Story 1 together:
Task: "Implement RAG orchestration service in server/services/rag.py"
Task: "Create OpenAI Agents SDK 'Book Expert' agent implementation in server/services/rag.py"
```

---

## Implementation Strategy

### MVP First (User Stories 1 and 2 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. Complete Phase 4: User Story 2
5. **STOP and VALIDATE**: Test User Stories 1 and 2 independently
6. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Add Health Check → Test independently → Deploy/Demo
6. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
   - Developer B: User Story 2
   - Developer C: User Story 3
   - Developer D: Health Check
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence