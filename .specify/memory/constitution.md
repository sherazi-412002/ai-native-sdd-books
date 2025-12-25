# The Autonomous Humanoid: A Comprehensive Guide Constitution (v1.1)

## Core Principles

### I. Phased Development Approach
Development follows three sequential phases: 
- Phase 1: Content & Platform (Docusaurus + 4-Module curriculum).
- Phase 2: Intelligence Layer (RAG with OpenAI ChatKit + Cohere Embeddings).
- Phase 3: Personalization & Auth (Better-Auth + i18n Urdu Translation).

### II. Hierarchical Content Integrity
The book must maintain a strict three-tier hierarchy: **Modules -> Chapters -> Sub-chapters**. Every chapter must be designed with clear front-matter to support the Docusaurus sidebar and the i18n translation engine.

### III. Intelligence & Embedding Strategy
The RAG system will use a "Hybrid Intelligence" stack:
- **Orchestration/UI:** OpenAI ChatKit SDK (for the chat interface and agent logic).
- **Embeddings:** Cohere API (specifically `embed-multilingual-v3.0` for English/Urdu compatibility).
- **Storage:** Qdrant Cloud (Vector) and Neon (Relational/History).

### IV. Multilingual & i18n Standard
Translation must not be a "hardcoded toggle." It must utilize the **Official Docusaurus i18n mechanism**. 
- Default locale: `en` (English).
- Target locale: `ur` (Urdu).
- Layout: RTL (Right-to-Left) support must be native to the Urdu locale.

### V. Dynamic Adaptation
Content personalization based on the user's "Software/Hardware Background" (captured via Better-Auth onboarding) should be implemented using React-based conditional rendering components within the Docusaurus MDX files.

---

## Technical Architecture (Refined)
- **Frontend:** Docusaurus (React), **i18n plugin**, GitHub Pages.
- **Backend:** FastAPI, OpenAI ChatKit SDK, Cohere.
- **Databases:** Neon Serverless Postgres, Qdrant Cloud.
- **Auth:** Better-Auth.

## Governance
Sequential execution is mandatory. **Any shift in embedding models or translation logic requires a new ADR.**

**Version**: 1.1.0 | **Ratified**: 2025-12-22
