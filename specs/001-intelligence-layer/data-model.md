# Data Model: Intelligence Layer - RAG Backend for Docusaurus Chatbot

## Entity: ChatSession
**Description**: Represents a user's conversation session with the chatbot

**Fields**:
- `id`: UUID (Primary Key) - Unique identifier for the session
- `user_id`: String (Optional) - Identifier for authenticated users
- `created_at`: DateTime - Timestamp when session was created
- `updated_at`: DateTime - Timestamp when session was last updated
- `title`: String (Optional) - Auto-generated title for the session
- `metadata`: JSON (Optional) - Additional session metadata

**Validation Rules**:
- `id` must be a valid UUID
- `created_at` must be in the past
- `updated_at` must be >= `created_at`

## Entity: Message
**Description**: Represents an individual message in a conversation

**Fields**:
- `id`: UUID (Primary Key) - Unique identifier for the message
- `session_id`: UUID (Foreign Key) - References the associated chat session
- `role`: String (Enum: "user", "assistant") - The sender of the message
- `content`: Text - The actual message content
- `timestamp`: DateTime - When the message was created
- `metadata`: JSON (Optional) - Additional message metadata

**Validation Rules**:
- `session_id` must reference an existing ChatSession
- `role` must be either "user" or "assistant"
- `content` must not be empty
- `timestamp` must be in the past

## Entity: DocumentChunk
**Description**: Represents a segment of book content that has been processed and embedded

**Fields**:
- `id`: UUID (Primary Key) - Unique identifier for the chunk
- `title`: String - Title of the document section
- `module`: String - Module identifier where this content belongs
- `chapter_url`: String - URL reference to the original chapter
- `content`: Text - The actual content of the chunk
- `embedding`: Array of floats (1024 dimensions) - The vector embedding
- `created_at`: DateTime - When the chunk was created
- `updated_at`: DateTime - When the chunk was last updated
- `metadata`: JSON - Additional metadata for retrieval

**Validation Rules**:
- `id` must be a valid UUID
- `title` must not be empty
- `content` must not be empty
- `embedding` must have exactly 1024 dimensions
- `chapter_url` must be a valid URL format

## Relationships

### ChatSession → Message (One-to-Many)
- One ChatSession can have multiple Messages
- Messages are deleted when their parent ChatSession is deleted (CASCADE)

### DocumentChunk (Stored in Qdrant)
- DocumentChunks are stored in Qdrant vector database
- No direct foreign key relationships but referenced by content during RAG retrieval

## State Transitions

### ChatSession
- Created when user starts a new conversation
- Updated when new messages are added
- Can be soft-deleted by user request
- Automatically archived after inactivity period

### Message
- Created when user sends a message or system generates a response
- Immutable after creation (no updates allowed)
- Deleted when parent session is deleted

## Constraints

1. **Data Integrity**: All foreign key relationships must be maintained
2. **Size Limits**: Message content limited to 10,000 characters
3. **Embedding Dimensions**: All DocumentChunk embeddings must have exactly 1024 dimensions
4. **Timestamp Validation**: All timestamps must be in UTC
5. **Metadata Structure**: JSON metadata must follow predefined schema for consistency