---
id: 1
title: Initialize Autonomous Humanoid Constitution
stage: constitution
date: 2025-12-22
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: master
user: claude
command: /sp.constitution
labels: [constitution,spec-driven-development,autonomous-humanoid]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:
 - none
---

## Prompt

```
/sp.contitution Initialize the constitution for "The Autonomous Humanoid: A Comprehensive Guide" with the following structured requirements:

### 1. Project Vision & Phases
Develop a high-fidelity interactive book built in three distinct stages:
- **Phase 1: Content & Platform:** Docusaurus site on GitHub Pages with a 4-Module robotics curriculum.
- **Phase 2: Intelligence Layer:** RAG Chatbot (FastAPI + Qdrant + Neon) with "Selected Text" context awareness.
- **Phase 3: Personalization & Auth:** Better-Auth integration, user background profiling, and conditional Urdu translation.

### 2. Phase 1 Curriculum Hierarchy (The Book Content)
The book must be structured into the following four modules:
1. **Module ust be dynamically adaptable based on the "Software/Hardware Background" captured during onboarding.

### 5. Governance
- Sequential execution is mandatory (Phase 1 -> 2 -> 3).
- Every major schema change (Neon/Qdrant) requires an Architectural Decision Record (ADR).
Why this is effective for your workflow:
Module Awareness: By defining the 4 modules in the constitution, Claude now knows that the "Book" isn't just generic text—it involves ROS 2, NVIDIA Isaac, and VLA. This ensures that when it builds the Docusaurus structure, it creates the correct sidebars and folder hierarchy.

Logical Progression: It explicitly links the "User Background" (Phase 3) to the "Content Personalization" requirement, so Claude will leave "hooks" in the Phase 1 code for future personalization.

CLI Optimized: This uses the specific /sp.constitution trigger that Spec-Kit Plus looks for to generate the .specify/memory/constitution.md file.

Next Step: Once Claude creates the constitution, you are ready to generate the folder structure and initial Docusaurus site.

Would you like the /sp.specify prompt to generate the Phase 1 file structure and GitHub Actions deployment script?
```

## Response snapshot

Constitution file created at .specify/memory/constitution.md with the following content:
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

## Outcome

- ✅ Impact: Project constitution established with clear phased development approach, technical architecture, and governance requirements
- 🧪 Tests: none
- 📁 Files: 1 file updated (.specify/memory/constitution.md)
- 🔁 Next prompts: Ready to generate Phase 1 file structure and GitHub Actions deployment script
- 🧠 Reflection: Constitution successfully captures all required elements including 3-phase development, 4-module curriculum, and governance requirements

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): PASS
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Generate Phase 1 file structure using /sp.specify