# 🤖 AI Agents — Complete Deep Dive Reference (2025)

---

## 1. What IS an AI Agent?

An AI agent is a system that:
1. **Perceives** — receives input (text, data, tool results)
2. **Reasons** — uses an LLM to decide what to do
3. **Acts** — calls tools, writes files, browses the web, talks to APIs
4. **Loops** — observes the result, re-reasons, acts again until done

The key difference from a plain chatbot: **a chatbot answers; an agent does**.

---

## 2. Terminology Decoded

This is confusing because every company uses different words for the same things.

| Term | What it actually means |
|------|------------------------|
| **Tool** | A Python function the LLM can call (e.g., `search_web()`, `read_file()`) |
| **Plugin** | A *group* of related tools (OpenAI/Semantic Kernel term) |
| **Skill** | Old Microsoft term for Plugin — now deprecated, means same thing |
| **Function** | Same as Tool — the actual callable unit |
| **Action** | Same as Tool — used by some platforms (e.g., Zapier AI Actions) |
| **Agent** | The reasoning system that decides when/which tool to call |
| **Chain** | A fixed, linear sequence of LLM calls (LangChain term) |
| **Pipeline** | Same as Chain — fixed sequence |
| **Workflow** | A more complex, possibly branching sequence of steps |
| **Orchestrator** | The system managing which agent/tool runs next |

> **TL;DR**: Tools = Functions = Skills = Actions = Plugins. They all mean "a thing the agent can do."

---

## 3. The Anatomy of an Agent

Every agent, regardless of framework, has these layers:

```
┌─────────────────────────────────────────────┐
│                  USER INPUT                  │
└────────────────────┬────────────────────────┘
                     │
┌────────────────────▼────────────────────────┐
│           REASONING ENGINE (LLM)             │
│   GPT-4o / Claude / Gemini / Llama etc.      │
│   → Decides what to do next                  │
└──────┬──────────────────────────┬────────────┘
       │                          │
┌──────▼──────┐          ┌────────▼────────────┐
│   MEMORY    │          │    TOOL LAYER        │
│ Short-term  │          │ search / code / DB   │
│ Long-term   │          │ file / browser / API │
│ Episodic    │          └─────────┬────────────┘
└─────────────┘                    │
                         ┌─────────▼────────────┐
                         │   ORCHESTRATOR        │
                         │ LangGraph / CrewAI    │
                         │ AutoGen / raw loop    │
                         └──────────────────────┘
```

---

## 4. Memory Types — Detailed

| Type | Where stored | Lifespan | Example use |
|------|-------------|----------|-------------|
| **In-context (working)** | LLM's prompt window | Current session only | Chat history, current task state |
| **External short-term** | Redis, SQLite | Session / hours | Remembering what user said 10 mins ago |
| **External long-term** | Vector DB (ChromaDB, Pinecone) | Permanent | User preferences, past documents |
| **Episodic** | File/DB log of past actions | Permanent | "Last week the agent did X" |
| **Semantic** | Vector embeddings of knowledge | Permanent | Company knowledge base, FAQ |
| **Procedural** | Hardcoded logic / system prompt | Permanent | "Always format output as JSON" |

---

## 5. Tools — How They Actually Work

A tool is just a Python function with a description. The LLM reads the description and decides whether to call it.

```python
# Example: Defining a tool in LangChain
from langchain.tools import tool

@tool
def search_web(query: str) -> str:
    """Search the web for current information. Use this when you need
    up-to-date facts that may not be in your training data."""
    # your actual search logic here
    return results

@tool
def read_file(filepath: str) -> str:
    """Read the contents of a local file. Use when the user
    references a specific file on their system."""
    with open(filepath) as f:
        return f.read()
```

The framework serializes these into a **JSON schema** that gets sent to the LLM alongside the conversation. The LLM responds with which tool to call and what arguments to use.

### Common Tool Categories

| Category | Examples |
|----------|---------|
| **Search / Retrieval** | Tavily Search, SerpAPI, DuckDuckGo, Wikipedia |
| **Code Execution** | Python REPL, E2B sandbox, Docker containers |
| **File System** | Read/write files, list directories |
| **Databases** | SQL query, MongoDB, Redis |
| **Browser / Web** | Playwright, Selenium, Browser Use |
| **Communication** | Email (Gmail), Slack, Teams, Discord |
| **APIs** | Any REST API wrapped as a function |
| **Calendar/Productivity** | Google Calendar, Notion, Jira |
| **Memory** | Store/retrieve from vector DB |
| **Multi-modal** | Generate image (DALL-E), speech-to-text, OCR |

---

## 6. Reasoning Patterns

### ReAct (Reason + Act) — The Foundation
The most common agent loop pattern. The LLM alternates between:

```
Thought: I need to find the current price of Apple stock.
Action: search_web("Apple AAPL stock price today")
Observation: AAPL is currently $189.43 as of May 18, 2025.
Thought: I have the answer. I can respond now.
Final Answer: Apple's stock (AAPL) is currently $189.43.
```

### Chain-of-Thought (CoT)
The LLM is prompted to think step-by-step *before* answering. No tool calls — just better reasoning.

### Plan-and-Execute
1. **Planner agent** creates a multi-step plan upfront
2. **Executor agent** runs each step
3. Good for complex, long-horizon tasks

### Reflection / Self-Critique
The agent generates an answer, then a second pass critiques it and improves it. Used in coding agents heavily.

---

## 7. Frameworks — Deep Dive

### 7.1 LangChain
**What it is**: The foundational Python library. Provides building blocks: LLM wrappers, prompt templates, document loaders, text splitters, tool wrappers, chains, and basic agents.

**Best for**: Starting out. It has integrations for almost every LLM, vector DB, and tool.

**Key concepts**:
- `ChatOpenAI`, `ChatAnthropic` — LLM wrappers
- `PromptTemplate` — reusable prompts
- `RunnableSequence` — chaining steps (new LCEL syntax)
- `AgentExecutor` — basic ReAct agent runner

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o")
tools = [search_web, read_file]  # your tools

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
executor.invoke({"input": "What is the weather in Singapore today?"})
```

---

### 7.2 LangGraph
**What it is**: Built on top of LangChain. Models agent workflows as a **graph** (nodes + edges). The production standard in 2025.

**Best for**: Complex agents that need loops, branching, state persistence, and human-in-the-loop.

**Key concepts**:
- **State** — a shared TypedDict that all nodes read/write
- **Nodes** — Python functions that do work
- **Edges** — define which node runs next
- **Conditional edges** — dynamic branching based on state
- **Checkpointing** — save/restore state (enables pause + resume)

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def llm_node(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def tool_node(state: AgentState):
    # execute the tool the LLM requested
    ...

def should_continue(state: AgentState):
    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
        return "tools"
    return END

builder = StateGraph(AgentState)
builder.add_node("llm", llm_node)
builder.add_node("tools", tool_node)
builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", should_continue)
builder.add_edge("tools", "llm")  # loop back

graph = builder.compile()
```

---

### 7.3 CrewAI
**What it is**: Role-based multi-agent framework. You define a "crew" of agents, each with a role, goal, backstory, and tools.

**Best for**: Quick prototyping of multi-agent systems. Very readable and intuitive.

```python
from crewai import Agent, Task, Crew

researcher = Agent(
    role="Senior Research Analyst",
    goal="Uncover cutting-edge developments in AI",
    backstory="You are an expert at finding and synthesizing information.",
    tools=[search_web],
    llm=llm
)

writer = Agent(
    role="Tech Content Strategist",
    goal="Craft compelling blog posts about AI topics",
    backstory="You transform complex topics into engaging narratives.",
    llm=llm
)

research_task = Task(
    description="Research the latest MCP developments",
    agent=researcher,
    expected_output="A detailed report on MCP"
)

write_task = Task(
    description="Write a blog post based on the research",
    agent=writer,
    expected_output="A 500-word blog post"
)

crew = Crew(agents=[researcher, writer], tasks=[research_task, write_task])
result = crew.kickoff()
```

---

### 7.4 AutoGen (Microsoft)
**What it is**: Conversational multi-agent framework from Microsoft. Agents talk to each other in a chat-like interface. Excellent for code generation + execution workflows.

**Best for**: Iterative coding tasks, research, anything requiring back-and-forth critique.

**Key pattern**:
```
User Proxy Agent ←→ Assistant Agent
     ↑                    ↑
Executes code         Writes code
Runs tests            Fixes bugs
Reports results       Tries again
```

---

### 7.5 Semantic Kernel (Microsoft)
**What it is**: Microsoft's enterprise-grade SDK for building AI agents (.NET + Python). Highly structured with Plugins, Functions, Planners, and a Kernel orchestrator.

**Best for**: Enterprise .NET shops, Azure-heavy environments, structured plugin ecosystems.

**Key concepts**:
- **Kernel** — the central orchestrator
- **Plugin** — a collection of related functions (replaces old "Skills" term)
- **Semantic Function** — a reusable prompt template
- **Native Function** — regular C#/Python code the agent can call
- **Planner** — auto-generates a plan from a user goal

---

### 7.6 LlamaIndex
**What it is**: Primarily a data framework. Best-in-class for RAG pipelines, document ingestion, and retrieval. Also has agent capabilities.

**Best for**: When your agent needs to reason over large document sets (PDFs, databases, wikis).

---

## 8. MCP — Model Context Protocol

**Created by**: Anthropic (November 2024)  
**Analogy**: "USB-C for AI" — a universal connector between agents and tools/data

### The Problem MCP Solves

Before MCP, connecting N data sources to M agents = N×M custom integrations.  
With MCP: each data source builds one MCP Server. Each agent builds one MCP Client. Done.

### Architecture

```
┌─────────────────┐         ┌──────────────────┐
│   MCP Client    │◄───────►│   MCP Server     │
│ (your AI agent) │  JSON-  │ (your tool/data) │
│ Claude Desktop  │  RPC    │ GitHub, SQLite,  │
│ VS Code, etc.   │         │ File system, etc.│
└─────────────────┘         └──────────────────┘
       Host
```

### What an MCP Server Exposes

| Primitive | Description | Example |
|-----------|-------------|---------|
| **Tools** | Functions the agent can call | `query_database()`, `send_email()` |
| **Resources** | Data/files the agent can read | A PDF, a database row, a config file |
| **Prompts** | Reusable prompt templates | "Summarize this in bullet points" |

### Transport Methods

| Method | Use case |
|--------|----------|
| **stdio** | Local tools (same machine, subprocess) |
| **HTTP + SSE** | Remote servers, cloud-based tools |

### MCP Example: Building a simple MCP Server

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

server = Server("my-tool-server")

@server.list_tools()
async def list_tools():
    return [
        types.Tool(
            name="get_weather",
            description="Get current weather for a city",
            inputSchema={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "get_weather":
        city = arguments["city"]
        # fetch weather...
        return [types.TextContent(type="text", text=f"Weather in {city}: 28°C, sunny")]

# Run the server
stdio_server(server)
```

Any MCP-compatible client (Claude Desktop, VS Code Copilot, custom agent) can now use this tool automatically.

---

## 9. A2A — Agent-to-Agent Protocol

**Created by**: Google (April 2025)  
**Complements**: MCP (they work together, not against each other)

### MCP vs A2A — The Key Distinction

```
MCP = How agents talk to TOOLS (vertical)
A2A = How agents talk to other AGENTS (horizontal)
```

| | MCP | A2A |
|--|-----|-----|
| **Purpose** | Agent ↔ Tool/Data | Agent ↔ Agent |
| **Architecture** | Client-Server | Peer-to-Peer |
| **Discovery** | Pre-configured | Dynamic via "Agent Cards" |
| **Creator** | Anthropic | Google |

### Agent Cards (A2A)
Each agent publishes a JSON "Agent Card" describing:
- What it can do
- What inputs it accepts
- What outputs it produces
- How to reach it

Other agents discover these cards and delegate tasks dynamically.

### How They Work Together

```
User → Orchestrator Agent
           │
           ├─[A2A]→ Research Agent → [MCP]→ Web Search Server
           │                        → [MCP]→ Database Server
           │
           └─[A2A]→ Writing Agent  → [MCP]→ File System Server
                                    → [MCP]→ Email Server
```

---

## 10. RAG — Retrieval-Augmented Generation

RAG solves the core problem of LLMs: **they only know what they were trained on**.

### How RAG Works

```
Step 1: INDEXING (done once)
  Your docs (PDFs, wikis, DBs)
       ↓
  Chunk into pieces (~500 tokens each)
       ↓
  Run through Embedding Model → vectors (lists of numbers)
       ↓
  Store in Vector Database

Step 2: RETRIEVAL (at query time)
  User question
       ↓
  Embed the question → query vector
       ↓
  Find most similar chunks in vector DB (cosine similarity)
       ↓
  Return top K relevant chunks

Step 3: GENERATION
  Inject retrieved chunks into prompt as context
       ↓
  LLM generates answer grounded in your actual data
```

### Embedding Models

| Model | Provider | Notes |
|-------|----------|-------|
| `text-embedding-3-large` | OpenAI | Industry standard, paid |
| `text-embedding-3-small` | OpenAI | Cheaper, still good |
| `embed-english-v3.0` | Cohere | Great for multilingual |
| `nomic-embed-text` | Nomic | Free, open source |
| `mxbai-embed-large` | MixedBread | Open source, top performer |

### Vector Database Comparison

| DB | Best for | Hosting | Cost |
|----|---------|---------|------|
| **ChromaDB** | Local dev, prototyping | Local/self-hosted | Free |
| **Pinecone** | Production, managed | Cloud | Paid (free tier) |
| **Weaviate** | Hybrid search, enterprise | Cloud or self-hosted | Free tier |
| **Qdrant** | High performance, local | Local or cloud | Free/paid |
| **pgvector** | Already using Postgres | Your Postgres DB | Free |
| **FAISS** | In-memory, research | In-process | Free |

---

## 11. Protocols & Standards Stack

```
┌──────────────────────────────────────────────┐
│              YOUR APPLICATION                │
├──────────────────────────────────────────────┤
│         AGENT FRAMEWORK LAYER                │
│   LangGraph / CrewAI / AutoGen / SK          │
├──────────────────────────────────────────────┤
│         AGENT COMMUNICATION                  │
│   A2A Protocol (agent ↔ agent)               │
├──────────────────────────────────────────────┤
│         TOOL/RESOURCE ACCESS                 │
│   MCP Protocol (agent ↔ tools/data)          │
├──────────────────────────────────────────────┤
│         LLM API LAYER                        │
│   OpenAI / Anthropic / Google / Ollama       │
└──────────────────────────────────────────────┘
```

---

## 12. Observability & Production Tools

You cannot ship an agent to production without knowing what it's doing.

### Tracing & Monitoring

| Tool | What it does | Notes |
|------|-------------|-------|
| **LangSmith** | Trace every LLM call, tool use, token cost | Best-in-class for LangChain/LangGraph |
| **Langfuse** | Open source LangSmith alternative | Self-hostable |
| **Arize Phoenix** | Evaluation + tracing | Open source |
| **Helicone** | LLM proxy with logging | Simple, cheap |
| **Weights & Biases** | ML experiment tracking | Good for eval |

### What You Should Be Logging

- Every LLM call (input prompt, output, model, tokens used, latency, cost)
- Every tool call (which tool, what args, what returned, how long)
- Every state transition (for graph-based agents)
- Errors and retries
- Human feedback (thumbs up/down)

### Guardrails

| Layer | What to do |
|-------|-----------|
| **Input** | Validate/sanitize user input. Block prompt injections |
| **Output** | Check output format, filter harmful content |
| **Tool execution** | Least privilege — only give agent tools it needs |
| **Code execution** | Always run in a sandbox (E2B, Docker) |
| **Audit log** | Log every action for compliance |

Tools: **Guardrails AI**, **NeMo Guardrails** (NVIDIA), **LlamaGuard** (Meta)

---

## 13. LLM Choices for Agents

| Model | Provider | Best for | Cost |
|-------|----------|---------|------|
| **GPT-4o** | OpenAI | General purpose, strong tool use | $$$ |
| **GPT-4o-mini** | OpenAI | Cheap, fast, good enough | $ |
| **Claude Sonnet 3.5/4** | Anthropic | Coding, long context, reasoning | $$$ |
| **Claude Haiku** | Anthropic | Fast, cheap, good for simple tasks | $ |
| **Gemini 1.5 Pro** | Google | Huge context window (1M tokens) | $$ |
| **Llama 3.1 70B** | Meta (via Ollama) | Fully local, free, private | Free |
| **Mistral Large** | Mistral | European, strong reasoning | $$ |
| **DeepSeek V3** | DeepSeek | Cheap, surprisingly capable | $ |

---

## 14. Multi-Agent Patterns

### Supervisor / Worker
```
Supervisor Agent
├── Worker Agent A (Researcher)
├── Worker Agent B (Coder)
└── Worker Agent C (Reviewer)
```
The supervisor breaks down tasks and delegates. Workers report back.

### Sequential Pipeline
```
Agent 1 (Gather data) → Agent 2 (Analyze) → Agent 3 (Write report)
```
Output of each agent is the input of the next.

### Hierarchical
```
CEO Agent
├── Manager Agent 1
│   ├── Worker A
│   └── Worker B
└── Manager Agent 2
    ├── Worker C
    └── Worker D
```
For very complex, many-layered workflows.

### Debate / Critique
```
Agent 1 proposes solution
Agent 2 critiques it
Agent 1 revises
... iterate until agreement
```
Used for high-stakes decisions (legal, medical, finance).

---

## 15. Complete Learning Path

```
BEGINNER
─────────
Week 1-2: Raw API calls
  → Call OpenAI/Anthropic API directly
  → Understand prompt engineering
  → Add basic tool calling manually

Week 3-4: First real agent
  → Build a ReAct loop from scratch
  → Add 3-5 tools (search, file, calculator)
  → Handle errors and retries

INTERMEDIATE
─────────────
Month 2: LangChain + LangGraph
  → Learn LCEL (LangChain Expression Language)
  → Build a stateful agent with LangGraph
  → Add memory (short-term + long-term)

Month 3: RAG
  → Build a document Q&A agent
  → ChromaDB + local embeddings
  → Chunk strategies, retrieval tuning

ADVANCED
─────────
Month 4: Multi-agent
  → CrewAI for quick multi-agent prototypes
  → LangGraph for production multi-agent graphs
  → Add human-in-the-loop checkpoints

Month 5: Production
  → Add LangSmith/Langfuse tracing
  → Implement guardrails
  → Deploy as API (FastAPI + Docker)

Month 6: Protocols
  → Build an MCP server for your own tools
  → Explore A2A for agent-to-agent communication
  → Evaluate: LangSmith evaluators, LLM-as-judge
```

---

## 16. Quick Decision Guide

```
What are you building?
│
├── Simple Q&A / chatbot with memory?
│   → Raw API + LangChain
│
├── Agent that needs to call tools?
│   → LangChain AgentExecutor or LangGraph
│
├── Multi-step, complex workflow with loops?
│   → LangGraph (state machine)
│
├── Team of AI agents with roles?
│   → CrewAI (fast) or LangGraph (production)
│
├── Agent that needs your documents?
│   → LlamaIndex + ChromaDB (RAG)
│
├── Code-writing/debugging agent?
│   → AutoGen or LangGraph + code sandbox
│
├── Enterprise, Azure-based?
│   → Semantic Kernel
│
└── Exposing your tools to any AI?
    → Build an MCP Server
```

---

*Last updated: May 2025 | Saved to: C:\Users\CM\Desktop\agent\*
