# E.V.A — Evolving Virtual Autonomy

> An autonomous personal AI system designed to reason, plan, learn, and act on behalf of its user.

---

## Current Milestone

**Milestone 1C — Model Intelligence Foundation**

EVA features a complete foundation (1A), deterministic cognitive pipeline (1B), and a provider-neutral model intelligence architecture (1C).

The model intelligence foundation establishes:
- **Provider Abstraction** — Asynchronous `ModelProvider` contract supporting text and structured output generation.
- **Model Router** — Deterministic `ModelRouter` mapping tasks (`conversation`, `understanding`, `reasoning`, `structured_output`) to provider instances.
- **Provider-Neutral Models** — Strongly-typed `ModelRequest`, `ModelResponse`, `ModelUsage`, and `ModelMetadata`.
- **Mock Provider** — Deterministic local provider requiring zero network access or API keys.
- **Future Extension Points** — Clean integration points for OpenAI, Gemini, Claude, and local models without core redesign.

---

## Vision

EVA is intended to become a long-lived autonomous personal AI system with:

- **Cognitive Loop** — Perceive → Reason → Plan → Act → Observe
- **Persistent Memory** — Episodic, semantic, and procedural memory
- **Autonomous Agents** — Goal-directed agents that can execute multi-step plans
- **Tool Integration** — Extensible tool system for real-world actions
- **Safety First** — Every consequential action evaluated before execution
- **Multi-Provider** — Not locked to any single AI provider

These capabilities will be built incrementally across future milestones.

---

## Architecture Overview

```
eva/
├── config/          # Environment-based configuration (pydantic-settings)
├── core/
│   ├── models.py    # Domain models (Goal, Plan, Action, Event, etc.)
│   ├── enums.py     # Shared enumerations
│   ├── context/     # In-memory context management
│   ├── self_model/  # EVA's operational self-representation
│   ├── brain/       # Cognitive loop (future)
│   ├── reasoning/   # Reasoning engine (future)
│   ├── planner/     # Planning system (future)
│   ├── goals/       # Goal management (future)
│   ├── decision/    # Decision engine (future)
│   └── personality/ # Personality model (future)
├── providers/       # Model provider abstraction + mock
├── safety/          # Safety policy and risk evaluation
├── memory/          # Memory subsystem (future)
├── perception/      # Perception subsystem (future)
├── agents/          # Agent subsystem (future)
├── tools/           # Tool subsystem (future)
├── devices/         # Device subsystem (future)
├── events/          # Event subsystem (future)
├── api/             # API subsystem (future)
├── runtime.py       # System lifecycle management
├── cli.py           # Interactive CLI
├── logging.py       # Centralized structured logging
└── __main__.py      # Module entry point
```

---

## Requirements

- Python 3.12+
- No external services required for basic operation

---

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd E.V.A

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install EVA with development dependencies
pip install -e ".[dev]"

# Copy environment configuration
cp .env.example .env
```

---

## Environment Setup

Edit `.env` to configure EVA. All settings have sensible defaults:

| Variable | Default | Description |
|---|---|---|
| `EVA_ENV` | `development` | Environment (development/testing/production) |
| `EVA_LOG_LEVEL` | `INFO` | Logging level |
| `EVA_DEBUG` | `false` | Enable debug mode |
| `EVA_DEFAULT_PROVIDER` | `mock` | Default AI model provider |
| `EVA_DEFAULT_MODEL` | `mock-v1` | Default model identifier |
| `EVA_MODEL_PROVIDER` | `mock` | AI model provider (alias) |
| `EVA_MODEL_NAME` | `mock-v1` | Model identifier (alias) |

No API keys, databases, or external services are required for Milestone 1C.

---

## Running EVA

```bash
# Using the module entry point
python -m eva

# Or using the installed console script
eva
```

Expected output:

```
====================================
           E . V . A
   Evolving Virtual Autonomy
====================================

  Status:       ONLINE
  Environment:  development
  Provider:     mock
  Mode:         interactive
  Safety:       ACTIVE
  Memory:       NOT CONNECTED
  Tools:        NONE

  EVA is ready.

You: hello

EVA: Hello. EVA is online and ready.

You: exit
```

---

## Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=eva --cov-report=term-missing
```

---

## Development Principles

1. **Clean interfaces** — Dependency inversion and protocol-based abstractions
2. **No circular dependencies** — Strict module boundaries
3. **Provider agnostic** — Core domain models never depend on external SDKs
4. **Safety first** — Actions evaluated before execution
5. **Async ready** — Async interfaces where future I/O is expected
6. **Testable** — Every meaningful component has tests
7. **Extensible** — Interfaces designed to evolve, not break

---

## License

MIT
