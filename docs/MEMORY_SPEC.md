# EVA Memory Subsystem Specification

> **Document Version:** 1.0.0
> **Status:** Authoritative Foundation Specification
> **Milestone:** 1E.1 — Memory Architecture Specification
> **Target Subsystem:** `eva.memory`

---

## 1. Executive Overview & Architectural Principles

Memory is the foundational substrate that transforms EVA from a stateless reactive conversational loop into a persistent, continuous, and evolving personal autonomous artificial intelligence. Without persistent memory, an agent is bound to immediate context windows; with a structured memory architecture, EVA accumulates knowledge, learns preferences, recalls past executions, and maintains long-term cognitive continuity across reboots, sessions, and devices.

### 1.1 Core Principles

1. **Layered Cognitive Hierarchy**
   Memory is not a monolithic vector store. It is organized into distinct functional tiers mirroring human cognitive architectures: immediate working memory, episodic experiential logs, semantic factual knowledge, procedural playbooks, and persistent user preferences.
2. **Provider-Agnostic Representation**
   Memory representations are canonical, typed, and vendor-neutral. Embeddings and similarity metrics interface through EVA's Model Provider abstraction (`ModelProvider` / `ModelRouter`), preventing lock-in to specific embedding vendors or vector database engines.
3. **Rigorous Provenance & Epistemic Traceability**
   Every memory record maintains an immutable record of origin. EVA explicitly distinguishes between *observed facts* (direct user statements, tool results, system events) and *inferred hypotheses* (model deductions, heuristic generalizations).
4. **Deterministic Isolation & Project Scoping**
   Memories are partitioned by tenancy: global persona and user preferences span all contexts, whereas project-specific architecture, code structures, and task memories remain strictly sandboxed within their designated workspace boundaries.
5. **Decoupled Storage Engine Roles**
   Storage responsibilities are cleanly allocated according to operational access characteristics: Redis manages volatile working state and low-latency cache; PostgreSQL manages durable relational schemas, transactional provenance, and structured search; and `pgvector` provides dense semantic vector indexing natively within PostgreSQL to prevent distributed sync drift.
6. **Passive Foundation Before Autonomous Agency**
   Milestone 1E defines the data structures, lifecycle stages, storage schemas, and retrieval protocols. Autonomous consolidation, background dreaming, and forgetting mechanisms are formally specified as future interfaces and will not be prematurely executed.

---

## 2. Memory Types (Taxonomy & Hierarchy)

EVA defines seven distinct memory types. Each serves a unique cognitive role with dedicated lifetimes, storage tiers, and retrieval semantics.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            COGNITIVE ENGINE                                 │
└──────┬───────────────────────────────┬───────────────────────────────┬──────┘
       │                               │                               │
       ▼                               ▼                               ▼
┌──────────────┐               ┌──────────────┐               ┌──────────────┐
│Working Memory│               │Session Memory│               │Preference Mem│
│ (Transient)  │               │ (Ephemeral)  │               │ (Persistent) │
└──────┬───────┘               └──────┬───────┘               └──────┬───────┘
       │                              │                              │
       └──────────────────────┬───────┴──────────────────────────────┘
                              │
                              ▼
        ┌───────────────────────────────────────────┐
        │        LONG-TERM PERSISTENT MEMORY        │
        ├─────────────────────┬─────────────────────┤
        │  Episodic Memory    │   Semantic Memory   │
        │  (Autobiographical) │   (Factual Knowledge│
        ├─────────────────────┼─────────────────────┤
        │  Procedural Memory  │   Project Memory    │
        │  (Skills/Workflows) │   (Workspace Scope) │
        └─────────────────────┴─────────────────────┘
```

### 2.1 Taxonomy Matrix

| Memory Type | Cognitive Purpose | Primary Lifetime | Storage Tier | Retrieval Mechanism |
|---|---|---|---|---|
| **Working Memory** | Active scratchpad for the ongoing cognitive turn / execution step. | Single turn or task execution (seconds to minutes) | In-memory `ContextManager` / Redis volatile keys | Direct memory lookup by turn/step ID |
| **Session Memory** | Conversational flow, multi-turn interaction continuity, dialog history. | Active conversation session (hours to days) | Redis list/hash, backed by PostgreSQL session table | Sequential temporal window + session ID |
| **Episodic Memory** | Autobiographical record of past events, actions taken, tool results, and outcomes. | Long-term (months to years, subject to decay) | PostgreSQL relational tables + pgvector embedding index | Hybrid: Semantic vector search + temporal range + metadata filters |
| **Semantic Memory** | Extracted facts, generalized concepts, world knowledge, and entity attributes. | Indefinite (reinforced over time) | PostgreSQL entity-attribute tables + pgvector dense embeddings | Dense semantic vector similarity + PostgreSQL full-text search (tsvector/tsquery) |
| **Procedural Memory** | Actionable recipes, workflows, tool execution patterns, and verified plans. | Permanent (updated on revision) | PostgreSQL structured templates + pgvector goal embeddings | Goal similarity matching + capability/tool filter |
| **Preference Memory** | Explicit user directives, tone preferences, operational constraints, and rules. | Permanent (until explicitly overridden) | PostgreSQL persistent key-value store; cached in Redis | Deterministic key/category lookup scoped by user ID |
| **Project Memory** | Context, conventions, architectural decisions, and files tied to a workspace. | Workspace/Project lifecycle | PostgreSQL partitioned by `project_id` + pgvector with project filter | Hard filter: `project_id = X` + dense semantic vector search |

### 2.2 Deep Dive into Memory Types

#### 2.2.1 Working Memory
Working memory holds the ephemeral context required by `CognitiveEngine` during active reasoning: parsed user intent, immediate candidate plans, intermediate tool outputs, and verification checks. It is stored directly in Python runtime memory (`eva.core.context.Context`) and mirrored to Redis under TTL keys when asynchronous background tasks run. It is wiped or committed upon cognitive cycle completion.

#### 2.2.2 Session Memory
Session memory maintains the thread of interaction across multiple cognitive turns. It ensures EVA knows what was said three turns ago without re-querying long-term databases. When a session terminates, a consolidation pipeline extracts salient facts into episodic and semantic stores before the raw session buffer expires.

#### 2.2.3 Episodic Memory
Episodic memory records "what happened, when, with whom, and what resulted." Each episode documents:
- Initial trigger or user request.
- Plan executed and tools invoked.
- Final outcome, user feedback, and verification status.
Episodic memory enables EVA to answer questions like: *"Did we run this database migration last week?"* or *"What error occurred when we deployed the staging server yesterday?"*

#### 2.2.4 Semantic Memory
Semantic memory represents distilled knowledge decoupled from specific narrative events. While episodic memory remembers *"The user told me on Monday that their PostgreSQL database runs on port 5433"*, semantic memory stores the consolidated fact: `User.Infrastructure.PostgreSQL.Port = 5433 (confidence=0.98)`.

#### 2.2.5 Procedural Memory
Procedural memory stores operational competency. When EVA successfully formulates and executes a multi-step plan (e.g., configuring a test environment, building a release bundle, executing a git rebase flow), the generalized recipe is indexed in procedural memory. Future similar requests retrieve this verified procedure rather than generating a plan from scratch.

#### 2.2.6 Preference Memory
User preferences represent personal operating guidelines (e.g., *"Always use TypeScript over JavaScript"*, *"Never run production commands without interactive confirmation"*, *"Be concise and omit conversational filler"*). Preferences carry highest priority during reasoning and are automatically injected into system prompts during prompt assembly.

#### 2.2.7 Project Memory
Project memory isolates context specific to a particular repository, client, or directory. It indexes codebase structure, conventions, build instructions, and active feature milestones. Project memory prevents cross-project contamination (e.g., architectural conventions from Project A bleeding into Project B).

---

## 3. Canonical Memory Object Specification

All memories throughout EVA share a canonical conceptual data model, ensuring uniform serialization, storage, indexing, and governance.

```text
┌────────────────────────────────────────────────────────────────────────┐
│                             MemoryRecord                               │
├────────────────────────────────┬───────────────────────────────────────┤
│ memory_id: UUID                │ Unique identifier                     │
│ memory_type: MemoryType        │ WORKING | SESSION | EPISODIC | ...    │
│ content: str                   │ Canonical textual description         │
│ structured_payload: dict       │ Structured attributes/parameters      │
│ source: MemorySource           │ USER | OBSERVATION | TOOL | MODEL ... │
│ confidence: float (0.0 - 1.0)  │ Epistemic certainty                   │
│ importance: float (0.0 - 1.0)  │ Long-term cognitive value             │
│ project_id: str | None         │ Workspace sandbox boundary            │
│ session_id: str | None         │ Interaction session boundary          │
│ provenance: ProvenanceRecord   │ Complete causal lineage               │
│ embedding_ref: EmbeddingRef    │ Model name, vector ID, dimensionality │
│ access_metadata: AccessMeta    │ Timestamps, access counts, decay rate │
│ retention_meta: RetentionMeta  │ Expiration, tier, immutability flag   │
└────────────────────────────────┴───────────────────────────────────────┘
```

### 3.1 Field Definitions

```python
class MemoryType(str, Enum):
    WORKING = "working"
    SESSION = "session"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    PREFERENCE = "preference"
    PROJECT = "project"

class MemorySource(str, Enum):
    USER_STATEMENT = "user_statement"       # Direct explicit user instruction/assertion
    EVA_OBSERVATION = "eva_observation"     # Sensor/telemetry observation
    TOOL_RESULT = "tool_result"             # Direct output of verified tool invocation
    MODEL_INFERENCE = "model_inference"     # LLM deduction/summary (hypothetical/inferred)
    EXTERNAL_DOCUMENT = "external_document" # Document, file, or web page content
    SYSTEM_EVENT = "system_event"           # Runtime lifecycle signal

class ProvenanceRecord(BaseModel):
    source_type: MemorySource
    source_id: str | None = None            # Turn ID, Event ID, or Tool Execution ID
    model_id: str | None = None             # LLM used to extract or infer this memory
    raw_quote: str | None = None            # Verbatim user utterance or tool snippet
    derived_from: list[UUID] = Field(default_factory=list) # Parent memory IDs if synthesized

class EmbeddingReference(BaseModel):
    model: str                              # Embedding model (e.g., "text-embedding-3-small")
    dimensions: int                         # Vector dimensionality (e.g., 1536)
    vector_id: str | None = None            # ID in pgvector column or partition
    version: int = 1                        # Re-embedding schema version

class AccessMetadata(BaseModel):
    created_at: datetime
    updated_at: datetime
    last_accessed_at: datetime | None = None
    access_count: int = 0
    decay_rate: float = 0.01                # Half-life parameter for recency scoring

class RetentionMetadata(BaseModel):
    expires_at: datetime | None = None      # Explicit TTL for transient memories
    is_immutable: bool = False              # Cannot be pruned; cannot override safety policies
    is_archived: bool = False               # Soft-deleted / compressed into cold archive
    retention_tier: str = "active"          # "active", "consolidated", "archived"
```

---

## 4. Storage Architecture (Redis, PostgreSQL, pgvector)

EVA implements a tier-structured storage architecture that strictly separates concerns across three database roles.

```
                          ┌───────────────────────────┐
                          │    EVA Memory Manager     │
                          └──────┬─────────────┬──────┘
                                 │             │
                ┌────────────────┘             └────────────────┐
                ▼                                               ▼
     ┌───────────────────────┐                     ┌─────────────────────────┐
     │         Redis         │                     │       PostgreSQL        │
     │  (In-Memory Cache)    │                     │  (ACID Relational Core) │
     ├───────────────────────┤                     ├─────────────────────────┤
     │ • Active turns        │                     │ • Canonical records     │
     │ • Session buffers     │                     │ • Relational integrity │
     │ • Cached preferences  │                     │ • Provenance tables     │
     │ • Distributed locks   │                     │ • Full-text search      │
     └───────────────────────┘                     └────────────┬────────────┘
                                                                │
                                                                ▼
                                                   ┌─────────────────────────┐
                                                   │        pgvector         │
                                                   │  (Vector Extension)     │
                                                   ├─────────────────────────┤
                                                   │ • Dense vector storage  │
                                                   │ • HNSW indexing         │
                                                   │ • Cosine similarity     │
                                                   │ • Unified transactions  │
                                                   └─────────────────────────┘
```

### 4.1 Responsibilities Matrix

#### 4.1.1 Redis (Tier 1 — Ephemeral & Real-Time)
- **Active Context Scratchpad**: Turn-level working context and pending goal states.
- **Session Ring Buffers**: Sliding window of the last $N$ turns per active session.
- **Hot-Cache for Preferences**: Frequently accessed user preference keys, avoiding database roundtrips during prompt compilation.
- **Coordination**: Distributed locks (`Redlock`) to prevent race conditions during memory consolidation.

#### 4.1.2 PostgreSQL (Tier 2 — Durable Relational System of Record)
- **Durable Memory Ledger**: Persistent home of all episodic, semantic, procedural, preference, and project records.
- **Relational Integrity**: Foreign key enforcement linking memories to users, sessions, goals, plans, and actions.
- **Provenance & Auditing**: Immutable audit trail documenting source, model inference chains, and modifications.
- **Deterministic Filtering**: High-performance B-tree indexed queries on `project_id`, `memory_type`, `created_at`, `confidence`, and `importance`.
- **Full-Text Search**: PostgreSQL full-text search using `tsvector`/`tsquery` with GIN indexes for exact keyword, function name, and token matching. BM25-style ranking may be introduced later through an appropriate extension or ranking implementation.

#### 4.1.3 pgvector (Tier 3 — Native Dense Vector Indexing)
- **Co-located Vector Embeddings**: Embeddings reside directly in PostgreSQL table columns (`embedding vector(1536)`).
- **Zero Synchronization Lag**: Because vectors live inside PostgreSQL, memory updates and vector updates occur in a single atomic ACID transaction. No dual-write drift or secondary sync failures.
- **HNSW Indexing**: Hierarchical Navigable Small World (HNSW) index targeting a benchmark-dependent low-latency retrieval target for nearest-neighbor semantic search (dependent on dataset scale, vector dimensionality, and host hardware resources).
- **Compound Relational Filtering**: Seamlessly combines vector distance with relational metadata in a single SQL query:
  ```sql
  SELECT memory_id, content, 1 - (embedding <=> :query_vec) AS similarity
  FROM memory_records
  WHERE (project_id = :project_id OR project_id IS NULL)
    AND memory_type = 'semantic'
    AND confidence >= 0.7
  ORDER BY embedding <=> :query_vec
  LIMIT 10;
  ```

---

## 5. Memory Lifecycle Pipeline

The memory subsystem operates as a continuous cognitive pipeline:

```text
Raw Sensory Input / Tool Result / Agent Action
                     │
                     ▼
             1. CAPTURE
             Store immediate event in Working Memory buffer
                     │
                     ▼
            2. NORMALIZE
             Strip formatting noise, resolve relative timestamps, tokenize
                     │
                     ▼
            3. CLASSIFY
             Categorize into candidate memory type (Episodic, Semantic, Preference)
                     │
                     ▼
            4. EVALUATE
             Assign Epistemic Confidence and Cognitive Importance
                     │
                     ▼
             5. STORE
             Write to PostgreSQL, compute embedding via pgvector, cache in Redis
                     │
                     ▼
            6. RETRIEVE
             Assemble relevant context for upcoming cognitive cycles
                     │
                     ▼
            7. REINFORCE
             Update access counts, recency timestamps, and confidence scores
                     │
                     ▼
           8. CONSOLIDATE (Future)
             Synthesize episodes into semantic facts; resolve contradictions
                     │
                     ▼
      9. RETAIN / ARCHIVE / FORGET (Future)
             Enforce TTLs, compress cold memories, prune low-importance records
```

### 5.1 Lifecycle Phase Specifications

1. **Capture**: Working memory captures raw input from user prompts, tool outputs, and action verifications without blocking immediate execution.
2. **Normalize**: Relative time expressions (*"yesterday"*, *"in two hours"*) are resolved to absolute UTC ISO-8601 timestamps. Text is sanitized and normalized.
3. **Classify**: Assigns incoming experiences to their target memory types (`Working`, `Session`, `Episodic`, `Semantic`, `Procedural`, `Preference`, `Project`). This classification is orchestrated by `MemoryManager` (or delegated to a dedicated future `MemoryClassifier` component):
   - Transient execution state and turn scratches $\rightarrow$ `Working Memory`
   - Active multi-turn conversational dialog flow $\rightarrow$ `Session Memory`
   - Concrete events, tool outcomes, and narrative actions $\rightarrow$ `Episodic Memory`
   - Extracted concepts, domain facts, and entity attributes $\rightarrow$ `Semantic Memory`
   - Verified workflows, execution recipes, and skill steps $\rightarrow$ `Procedural Memory`
   - Explicit user directives, operating constraints, and tone choices $\rightarrow$ `Preference Memory`
   - Workspace conventions, repo structure, and project context $\rightarrow$ `Project Memory`
4. **Evaluate**: Calculates confidence (source certainty) and importance (persistence value).
5. **Store**: Writes transactional records to PostgreSQL, computes dense embeddings, updates `pgvector` indexes, and caches hot entries in Redis.
6. **Retrieve**: Multi-stage hybrid search combines semantic similarity, keyword matching, and relational filters to supply context to `CognitiveEngine`.
7. **Reinforce**: Successfully retrieved memories that contribute to validated action plans receive an access counter increment and recency boost.
8. **Consolidate** *(Future Interface)*: Background tasks periodically aggregate episodic logs into generalized semantic knowledge units and identify outdated facts.
9. **Retain / Archive / Forget** *(Future Interface)*: Enforces retention policies, soft-deletes obsolete records, and moves stale episodes to compressed archival storage.

---

## 6. Retrieval Architecture (Hybrid Multi-Stage Search)

EVA does not rely exclusively on vector search. Dense vector search is prone to semantic drift and struggles with exact keyword queries (such as function names, error codes, and specific IDs). EVA's retrieval architecture implements a five-stage hybrid pipeline:

```text
User Query / Task Context
           │
           ▼
Stage 1: Pre-Filtering (Hard Constraints)
         • project_id = :active_project OR project_id IS NULL
         • memory_type IN (:required_types)
         • confidence >= :min_confidence
           │
           ▼
Stage 2: Parallel Candidate Retrieval
         ├─ Branch A: Dense Vector Search (pgvector HNSW)
         ├─ Branch B: Exact Keyword Search (PostgreSQL full-text search using tsvector/tsquery)
         └─ Branch C: Deterministic Cache Lookup (Redis Preferences)
           │
           ▼
Stage 3: Reciprocal Rank Fusion (RRF) & Scoring
         Combine dense and sparse ranks with weights
           │
           ▼
Stage 4: Cognitive Modulation
         Modulate scores with Importance and Temporal Recency Decay
           │
           ▼
Stage 5: Context Budget Assembly & Deduplication
         Fit top-ranked memories within prompt token budget
```

### 6.1 Composite Ranking Formula

For candidate memory $m$ matching query $q$:

$$Score(m, q) = w_{sem} \cdot S_{semantic}(m, q) + w_{keyword} \cdot S_{keyword}(m, q) + w_{imp} \cdot m.importance + w_{rec} \cdot e^{-\lambda \cdot \Delta t}$$

Where:
- $S_{semantic}(m, q) \in [0, 1]$: Cosine similarity between query embedding and memory embedding.
- $S_{keyword}(m, q) \in [0, 1]$: Normalized PostgreSQL full-text search using `tsvector`/`tsquery`. BM25-style ranking may be introduced later through an appropriate extension or implementation.
- $m.importance \in [0, 1]$: Inherent value assigned during evaluation.
- $\Delta t$: Elapsed time since last update / access in days.
- $\lambda$: Temporal decay rate ($0.01$ default; $0.0$ for immutable memories).
- Weights default to: $w_{sem} = 0.45$, $w_{keyword} = 0.25$, $w_{imp} = 0.15$, $w_{rec} = 0.15$.

---

## 7. Memory Component Interfaces (Conceptual Architecture)

To maintain strict provider-neutrality and decoupling from external frameworks, the memory subsystem defines clean Python Protocol abstractions. These interfaces define the contracts that will be implemented in subsequent milestones.

```python
from typing import Protocol, Sequence
from uuid import UUID

class MemoryReader(Protocol):
    """Protocol for memory retrieval operations."""
    async def get_by_id(self, memory_id: UUID) -> MemoryRecord | None: ...
    async def query(self, request: MemoryQueryRequest) -> list[ScoredMemoryRecord]: ...
    async def get_preferences(self, user_id: str) -> dict[str, Any]: ...

class MemoryWriter(Protocol):
    """Protocol for memory persistence and mutation."""
    async def store(self, record: MemoryRecord) -> UUID: ...
    async def store_batch(self, records: Sequence[MemoryRecord]) -> list[UUID]: ...
    async def update(self, memory_id: UUID, updates: dict[str, Any]) -> bool: ...
    async def delete(self, memory_id: UUID) -> bool: ...
    async def touch(self, memory_id: UUID) -> None: ...

class MemoryConsolidator(Protocol):
    """Protocol for future background memory consolidation."""
    async def consolidate_session(self, session_id: str) -> list[MemoryRecord]: ...
    async def summarize_episodes(self, project_id: str | None, time_range: tuple[datetime, datetime]) -> list[MemoryRecord]: ...
    async def resolve_contradictions(self, records: Sequence[MemoryRecord]) -> list[MemoryRecord]: ...

class MemoryRetentionPolicy(Protocol):
    """Protocol for memory retention, archival, and forgetting."""
    async def evaluate_retention(self) -> RetentionReport: ...
    async def archive_stale(self, older_than: datetime) -> int: ...
    async def purge_expired(self) -> int: ...

class MemoryClassifier(Protocol):
    """Protocol for classifying incoming experiences into memory types.

    Responsible for categorizing incoming content into Working, Session,
    Episodic, Semantic, Procedural, Preference, or Project memory types.
    This classification responsibility is held by MemoryManager directly
    or delegated to a dedicated MemoryClassifier component.
    """
    async def classify(self, content: str, context: Context) -> MemoryType: ...

class MemoryManager(Protocol):
    """Primary facade coordinating memory classification, readers, writers, and policies.

    Holds primary responsibility for classifying incoming records into appropriate
    memory types (Working, Session, Episodic, Semantic, Procedural, Preference,
    Project), either natively or via a delegated MemoryClassifier component.
    """
    @property
    def reader(self) -> MemoryReader: ...
    @property
    def writer(self) -> MemoryWriter: ...
    @property
    def classifier(self) -> MemoryClassifier | None: ...
    async def assemble_context(self, task_context: Context) -> MemoryContextPayload: ...
```

---

## 8. Provenance & Epistemic Certainty

EVA explicitly tracks the origin and confidence of every stored piece of information. This prevents "hallucination amplification," where an AI models an uncertain deduction as an absolute ground truth.

### 8.1 Source Hierarchy and Initial Confidence

```text
┌──────────────────────┬──────────────────────┬───────────────────────────────┐
│ Memory Source        │ Initial Confidence   │ Epistemic Status              │
├──────────────────────┼──────────────────────┼───────────────────────────────┤
│ USER_STATEMENT       │ 0.95 - 1.00          │ Direct User Claim / Directive │
│ TOOL_RESULT          │ 0.90 - 1.00          │ Verified Operational Fact     │
│ EVA_OBSERVATION      │ 0.85 - 0.95          │ Direct System Telemetry       │
│ EXTERNAL_DOCUMENT    │ 0.70 - 0.85          │ External Claim                │
│ MODEL_INFERENCE      │ 0.50 - 0.75          │ Inferred Hypothesis           │
│ SYSTEM_EVENT         │ 0.95 - 1.00          │ Deterministic Runtime Event   │
└──────────────────────┴──────────────────────┴───────────────────────────────┘
```

> **Illustrative Priors Note:** The source-based confidence ranges listed in the table above represent illustrative priors, not fixed confidence values. In operational practice, initial confidence values are dynamically calibrated based on contextual verification, source fidelity, and corroboration.

### 8.2 Provenance Chain
When a semantic fact is synthesized from multiple past episodes:
1. The new `MemoryRecord` references the IDs of all contributing memories in `provenance.derived_from`.
2. The synthesized confidence is bounded by:
   $$\text{Confidence}_{\text{synth}} = \min(\text{Confidence}_{\text{parents}}) \times \text{Confidence}_{\text{inference}}$$
3. If a parent memory is subsequently refuted by direct user correction, downstream synthesized memories are automatically flagged for re-verification.

---

## 9. Confidence, Importance, and Relevance

EVA distinguishes between three fundamental cognitive metrics that are frequently conflated:

### 9.1 Confidence (Epistemic Certainty)
- **Definition**: The probability that the memory represents objective truth or verified reality.
- **Scale**: $0.0$ (complete speculation) to $1.0$ (maximum empirical certainty). Source-based confidence ranges serve as illustrative priors rather than fixed confidence values.
- **Dynamics**: Increases when corroborated by independent tool results or user confirmation; decreases when contradicted by newer evidence.

### 9.2 Importance (Long-Term Cognitive Significance)
- **Definition**: The inherent value of retaining this memory for future goals, safety constraints, and user alignment.
- **Scale**: $0.0$ (trivial conversational noise, e.g., *"thanks"*) to $1.0$ (critical safety rule or core architectural constraint).
- **Dynamics**: Static over time unless explicitly re-evaluated by consolidation policies. High-importance memories ($> 0.8$) are exempt from automated forgetting.

### 9.3 Relevance (Dynamic Situational Utility)
- **Definition**: How useful a specific memory is to the *current task or query*.
- **Scale**: Computed dynamically during retrieval using semantic similarity, keyword overlap, and goal alignment.
- **Dynamics**: Fluctuates per cognitive turn. A memory with $1.0$ importance and $1.0$ confidence may have $0.0$ relevance to a completely unrelated turn.

---

## 10. Project and Context Isolation

To serve developers and knowledge workers across diverse contexts, EVA enforces strict workspace sandboxing.

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        GLOBAL SCOPE (All Projects)                     │
│  • User identity and core persona                                      │
│  • General programming knowledge & language skills                     │
│  • Global user preferences (tone, security boundaries)                 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
           ┌────────────────────────┴────────────────────────┐
           ▼                                                 ▼
┌─────────────────────────────────────┐   ┌─────────────────────────────────────┐
│      PROJECT A SCOPE (e.g., E.V.A)  │   │     PROJECT B SCOPE (e.g., Portal)  │
│  • Repo architecture & file map     │   │  • Domain models & API endpoints    │
│  • Specific dependency choices      │   │  • Specific deployment procedures   │
│  • Active milestones & branch state │   │  • Client-specific conventions      │
└─────────────────────────────────────┘   └─────────────────────────────────────┘
```

### 10.1 Scoping Rules
1. **Global Isolation**: A project-scoped memory (`project_id = "eva"`) is **never** visible to queries originating from a different workspace (`project_id = "portal"`).
2. **Inheritance**: Queries within a project context automatically receive both project-specific memories AND global preferences/skills:
   $$\text{Scope}_{\text{visible}} = \{m \in \text{Memories} \mid m.\text{project\_id} = \text{active\_project} \lor m.\text{project\_id} \text{ IS NULL}\}$$
3. **Collision Resolution**: In the event of a conflict between a global preference and a project-specific directive, the project directive takes precedence within that project's scope.

---

## 11. Privacy, Safety, and Governance

Because EVA operates as an intimate personal assistant with access to local systems, the memory foundation incorporates robust data protection controls:

1. **Memory Access Boundaries**
   Memory inspection and management interfaces require authorized local session tokens. External API endpoints or unauthenticated sub-processes cannot read memory records.
2. **Sensitive Data Sanitization**
   Raw passwords, API keys, private tokens, and sensitive credentials must never be committed to persistent memory. If a tool output or user prompt contains recognized credentials, they are sanitized/redacted prior to storage.
3. **User Right to Erasure (`forget`)**
   The user retains absolute sovereignty over EVA's memory. Explicit commands such as `"forget our discussion about X"` or `"delete all project memories for Y"` execute cascade soft-deletes and purge corresponding embeddings from `pgvector`.
4. **Safety Directive Immutability & Independence**
   Core safety policies (e.g., constraints prohibiting unconfirmed execution of destructive file operations) carry `is_immutable = True` and cannot be modified or eroded by adversarial user prompts or prompt injections. Protected and immutable memory records cannot override EVA's independent safety and authorization policies; safety evaluation and authorization policies operate independently of memory content and always take strict precedence.
5. **Full Auditability**
   Every memory modification, access, reinforcement, and deletion is recorded in a transactional audit log containing timestamp, caller ID, and reason.

---

## 12. Future Extensions Roadmap

The Milestone 1E foundation is designed to support the following advanced capabilities in future milestones without requiring structural schema redesigns:

1. **Multimodal Memory Support** (Milestone 2.x)
   Extension of `content` and vector embeddings to support visual snapshots (UI state, diagram images) and audio waveforms via multimodal embedding models.
2. **Knowledge Graph Integration** (Milestone 2.x)
   Augmenting pgvector with an entity-relationship graph (using Apache AGE or native PostgreSQL recursive CTEs) to support multi-hop relational graph traversal.
3. **Autonomous Memory Consolidation ("Dreaming")** (Milestone 3.x)
   Off-peak background jobs that cluster episodes, synthesize higher-level abstractions, resolve contradictory claims, and calculate memory decay.
4. **Cross-Device Encrypted Synchronization** (Milestone 4.x)
   End-to-end encrypted synchronization protocol replicating memory records between a user's desktop, server, and mobile nodes.
5. **Learned Retrieval Modulation** (Milestone 5.x)
   Reinforcement learning algorithms adjusting retrieval weights ($w_{sem}, w_{keyword}, w_{imp}, w_{rec}$) based on whether retrieved memories successfully aided plan execution.

---

## 13. Summary & Roadmap Alignment

| Milestone | Scope | Deliverables |
|---|---|---|
| **1E.1** *(Current)* | **Memory Specification** | Authoritative architecture, data models, storage definitions, and lifecycle contracts (`docs/MEMORY_SPEC.md`). |
| **1E.2** | **Memory Data Models & In-Memory Store** | Typed Pydantic models in `eva/memory/models.py`, in-memory repository for zero-dependency local testing. |
| **1E.3** | **Relational Storage & Redis Integration** | PostgreSQL schema migration, Redis session/working memory cache. |
| **1E.4** | **Vector Search with pgvector** | Embedding generation via `ModelProvider`, HNSW index setup, hybrid retrieval pipeline. |
| **1E.5** | **Cognitive Engine Memory Integration** | Integration of `MemoryManager` into `CognitiveEngine` for prompt injection and experience capture. |
