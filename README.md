# 🚀 Kasparro AI - True Multi-Agent Content Generation System

A **true multi-agent system** that transforms raw, unstructured product data into production-ready, structured content. Unlike simple sequential pipelines, this system features **autonomous agents**, **dynamic coordination**, and **message-based communication**.

---

## 📋 Table of Contents

1. [Overview](#1-overview)
2. [Multi-Agent Architecture](#2-multi-agent-architecture)
3. [Quick Start](#3-quick-start)
4. [Project Structure](#4-project-structure)
5. [Core Framework](#5-core-framework)
6. [Agents](#6-agents)
7. [Running Tests](#7-running-tests)
8. [Logging & Observability](#8-logging--observability)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Overview

This system implements a **True Multi-Agent Architecture** demonstrating:

| Feature | Description |
|---------|-------------|
| **Agent Autonomy** | Each agent has goals, state, memory, and decision-making |
| **Dynamic Coordination** | Orchestrator assigns tasks based on capabilities |
| **Message-Based Communication** | Agents communicate via message bus |
| **Shared Knowledge** | Blackboard pattern for data exchange |
| **Dependency Resolution** | Tasks execute respecting dependencies |

---

## 2. Multi-Agent Architecture

```
                    ┌──────────────────────────────────────┐
                    │           ORCHESTRATOR               │
                    │  • Agent Registry                    │
                    │  • Task Queue & Scheduling           │
                    │  • Dependency Resolution             │
                    │  • Capability-Based Routing          │
                    └──────────────┬───────────────────────┘
                                   │
            ┌──────────────────────┼──────────────────────┐
            │                      │                      │
            ▼                      ▼                      ▼
  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
  │  PARSER AGENT   │  │ STRATEGY AGENT  │  │  BUILDER AGENT  │
  │                 │  │                 │  │                 │
  │ • parse_raw_data│  │ • gen_competitor│  │ • build_product │
  │ • validate_data │  │ • generate_faqs │  │ • build_faq     │
  │                 │  │                 │  │ • build_compare │
  │ State: IDLE →   │  │ State: IDLE →   │  │ State: IDLE →   │
  │ THINKING →      │  │ THINKING →      │  │ THINKING →      │
  │ EXECUTING       │  │ EXECUTING       │  │ EXECUTING       │
  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
           │                    │                    │
           └────────────────────┼────────────────────┘
                                │
              ┌─────────────────┴─────────────────┐
              │                                   │
        ┌─────▼─────┐                     ┌───────▼───────┐
        │  MESSAGE  │                     │  BLACKBOARD   │
        │    BUS    │                     │ (Shared State)│
        │           │                     │               │
        │ • Pub/Sub │                     │ • product_data│
        │ • Routing │                     │ • competitor  │
        │ • History │                     │ • faq_qs      │
        └───────────┘                     │ • pages       │
                                          └───────────────┘
```

### Key Differences from Sequential Pipeline

| Aspect | Old (Sequential) | New (Multi-Agent) |
|--------|------------------|-------------------|
| Control Flow | Hardcoded in main.py | Dynamic via Orchestrator |
| Communication | Direct function calls | Message Bus |
| Data Sharing | Return values | Shared Blackboard |
| Task Assignment | Fixed order | Capability-based routing |
| Agent State | None | Full state machine |
| Decision Making | Centralized | Distributed to agents |
| Collaboration | None | Request assistance pattern |

---

## 3. Quick Start

### Prerequisites

- **Python 3.10+**
- **Google Gemini API Key** ([Get one here](https://aistudio.google.com/apikey))

### Step 1: Clone & Install

```bash
# Clone the repository
git clone <your-repo-url>
cd kasparro-agentic-Varun-KS

# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure API Key

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

### Step 3: Run the Multi-Agent System

```bash
# Windows PowerShell
$env:PYTHONPATH = "."
python -m src.main

# Linux/Mac
PYTHONPATH=. python -m src.main
```

### Step 4: Check Output

Generated files in `output/`:
- `product_page.json` - Product information page
- `faq.json` - FAQ with accurate answers
- `comparison_page.json` - Product vs competitor

---

## 4. Project Structure

```
kasparro-agentic-Varun-KS/
│
├── src/
│   ├── core/                      # 🆕 Multi-Agent Framework
│   │   ├── __init__.py
│   │   ├── base_agent.py         # Abstract autonomous agent
│   │   ├── messages.py           # Message bus & protocol
│   │   ├── blackboard.py         # Shared knowledge space
│   │   └── orchestrator.py       # Dynamic coordinator
│   │
│   ├── agents/                    # Autonomous Agents
│   │   ├── parser_agent.py       # Data extraction agent
│   │   ├── strategy_agent.py     # Strategic planning agent
│   │   └── builder_agent.py      # Content assembly agent
│   │
│   ├── blocks/                    # Reusable logic blocks
│   │   ├── benefits.py
│   │   ├── usage.py
│   │   └── comparison.py
│   │
│   ├── models/                    # Pydantic data models
│   │   ├── internal.py
│   │   └── output.py
│   │
│   ├── utils/
│   │   ├── llm_client.py
│   │   └── logger.py
│   │
│   └── main.py                    # Multi-agent entry point
│
├── data/
│   └── raw_input.txt              # Input data
│
├── output/                        # Generated JSON files
├── logs/                          # Execution logs
├── docs/
│   └── projectdocumentation.md   # Detailed architecture docs
│
├── tests/
│   └── test_integrity.py
│
├── requirements.txt
└── README.md
```

---

## 5. Core Framework

### BaseAgent (`src/core/base_agent.py`)

Abstract class providing agent autonomy:

```python
class BaseAgent(ABC):
    # Identity & Capabilities
    agent_id: str
    capabilities: List[AgentCapability]
    
    # State Machine
    state: AgentState  # IDLE, THINKING, EXECUTING, WAITING, COMPLETED
    
    # Memory (decisions, outcomes, observations)
    memory: AgentMemory
    
    # Abstract methods - subclasses implement these
    def plan(goal) -> List[Action]   # Agent plans autonomously
    def execute(plan, goal) -> bool  # Agent executes plan
    
    # Communication
    def send_message(type, recipient, content)
    def post_to_blackboard(key, value)
    def request_assistance(task, capability)
```

### Orchestrator (`src/core/orchestrator.py`)

Dynamic coordination engine:

```python
class Orchestrator:
    registry: AgentRegistry   # Agent discovery
    message_bus: MessageBus   # Communication backbone
    blackboard: Blackboard    # Shared state
    
    # Dynamic task assignment
    def submit_workflow(workflow)
        # 1. Analyze task dependencies
        # 2. Find capable agents
        # 3. Assign goals to agents
        # 4. Monitor progress
        # 5. Handle failures & collaboration
```

### Message Types (`src/core/messages.py`)

```python
class MessageType(Enum):
    TASK_REQUEST      # Request agent to do something
    TASK_COMPLETE     # Report completion
    DATA_REQUEST      # Request data from agent
    DATA_RESPONSE     # Respond with data
    GOAL_ASSIGNED     # Orchestrator assigns goal
    NEED_ASSISTANCE   # Request help from others
```

---

## 6. Agents

### Parser Agent
**Capabilities**: `parse_raw_data`, `validate_data`

**Autonomous Behavior**:
- Decides how to read input (file vs provided text)
- Creates multi-step plan for extraction
- Validates output before publishing to blackboard

### Strategy Agent
**Capabilities**: `generate_competitor`, `generate_faqs`

**Autonomous Behavior**:
- Acquires product data from blackboard
- Decides which strategic tasks to perform
- Can request assistance if data missing

### Builder Agent
**Capabilities**: `build_product_page`, `build_faq_page`, `build_comparison_page`

**Autonomous Behavior**:
- Gathers required data from blackboard
- Coordinates logic blocks for content
- Requests missing data from other agents

---

## 7. Running Tests

```bash
# Windows PowerShell
$env:PYTHONPATH = "."
python -m pytest tests/test_integrity.py -v

# Linux/Mac
PYTHONPATH=. python -m pytest tests/test_integrity.py -v
```

---

## 8. Logging & Observability

### Console Output
```
INFO | KASPARRO AI CONTENT ENGINE - Pipeline Started
INFO | Initializing Multi-Agent System...
INFO | Orchestrator initialized
INFO | Registered 3 agents
INFO | Available capabilities: ['parse_raw_data', 'validate_data', ...]
INFO | Workflow defined: Content Generation Pipeline
INFO | Assigned task 'Parse Product Data' to Parser Agent
INFO | [OK] Product page saved
INFO | PIPELINE COMPLETE
```

### Agent Decision Trail
The system logs agent decisions:
```
Parser Agent Decisions:
  └─ Read from file
      Reasoning: File path provided: data/raw_input.txt
  └─ Execute plan for: Extract and structure product data
      Reasoning: Generated 4 steps
```

### Detailed Logs
```bash
type logs\system.log  # Windows
cat logs/system.log   # Linux/Mac
```

---

## 9. Troubleshooting

### `ModuleNotFoundError: No module named 'src'`
```bash
$env:PYTHONPATH = "."  # Windows PowerShell
export PYTHONPATH=.    # Linux/Mac
```

### `GEMINI_API_KEY not found`
Create `.env` file with your API key.

---

## 📚 Documentation

For detailed architecture documentation, see [`docs/projectdocumentation.md`](docs/projectdocumentation.md).

---
