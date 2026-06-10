# LangGraph · MCP · RAG — Agentic AI System

A production-ready agentic AI system that combines **LangGraph**, the **Model Context Protocol (MCP)**, and **Retrieval-Augmented Generation (RAG)** — fully containerised with Docker Compose.

> Built as a hands-on demonstration of the core patterns used in modern AI engineering: autonomous agents, tool orchestration, and grounded generation.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User (browser)                        │
│                     http://localhost:7860                    │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP + SSE (streaming)
┌──────────────────────────▼──────────────────────────────────┐
│                     Web Service (FastAPI)                    │
│          Streaming chat UI · /ask endpoint · SSE             │
└──────────────────────────┬──────────────────────────────────┘
                           │ LangGraph agentic loop
┌──────────────────────────▼──────────────────────────────────┐
│                  LangGraph ReAct Agent                       │
│     StateGraph · ToolNode · ChatOllama (Llama 3.2)          │
└──────────────────────────┬──────────────────────────────────┘
                           │ MCP protocol (SSE transport)
┌──────────────────────────▼──────────────────────────────────┐
│                   MCP Server (FastMCP)                       │
│   Tools: search_knowledge_base · add_document · stats       │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP
┌──────────────────────────▼──────────────────────────────────┐
│                  ChromaDB (vector store)                     │
│         Embeddings: all-MiniLM-L6-v2 (sentence-transformers)│
└─────────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    Ollama (LLM runtime)                      │
│                  Model: Llama 3.2 (local)                    │
└─────────────────────────────────────────────────────────────┘
```

### How it works

1. The user sends a question from the web interface
2. The **LangGraph ReAct agent** receives the question and decides whether to search the knowledge base
3. If needed, it calls the **MCP server** via the Model Context Protocol
4. The MCP server runs a **semantic search** against ChromaDB using vector embeddings
5. The retrieved documents are injected into the LLM context
6. **Llama 3.2** generates a grounded answer, streamed token-by-token back to the browser

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Agent framework** | [LangGraph](https://github.com/langchain-ai/langgraph) — StateGraph, ToolNode, ReAct loop |
| **Tool protocol** | [Model Context Protocol](https://modelcontextprotocol.io/) — FastMCP (SSE transport) |
| **LLM** | [Llama 3.2](https://ollama.com/library/llama3.2) via [Ollama](https://ollama.com/) — runs 100% locally |
| **Vector store** | [ChromaDB](https://www.trychroma.com/) — persistent embeddings |
| **Embeddings** | `all-MiniLM-L6-v2` via sentence-transformers |
| **Web backend** | [FastAPI](https://fastapi.tiangolo.com/) + SSE streaming |
| **Frontend** | Vanilla HTML/CSS/JS — zero framework dependencies |
| **Infra** | Docker Compose — multi-service, health checks, init containers |

---

## Key Concepts Demonstrated

### LangGraph — Stateful Agentic Loop
The agent is built as a `StateGraph` with two nodes (`agent` and `tools`) and a conditional edge that loops until the LLM decides no more tool calls are needed. This implements the [ReAct](https://arxiv.org/abs/2210.03629) pattern (Reasoning + Acting).

```python
graph = StateGraph(MessagesState)
graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode(tools))
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")
```

### Model Context Protocol (MCP)
The RAG capabilities are exposed as MCP tools, making them reusable by any MCP-compatible client. The agent connects to the MCP server over HTTP/SSE — the same transport used in production MCP deployments.

```python
@mcp.tool()
def search_knowledge_base(query: str, n_results: int = 4) -> str:
    """Search the knowledge base for documents relevant to the query."""
    ...
```

### RAG — Retrieval-Augmented Generation
Documents are chunked, embedded at indexing time, and retrieved by semantic similarity at query time. This grounds the LLM's responses in the knowledge base and reduces hallucinations.

### Streaming with SSE
Responses are streamed token-by-token from the LLM to the browser using Server-Sent Events, with word-boundary buffering to ensure clean output.

---

## Project Structure

```
.
├── docker-compose.yml          # Orchestration (5 services)
├── .env.example
│
├── web/                        # FastAPI chat interface
│   ├── app.py                  # SSE streaming endpoint
│   ├── static/index.html       # Chat UI
│   ├── Dockerfile
│   └── requirements.txt
│
├── agent/                      # LangGraph ReAct agent (CLI)
│   ├── main.py
│   ├── Dockerfile
│   └── requirements.txt
│
└── mcp_server/                 # FastMCP RAG server
    ├── server.py               # MCP tools + ChromaDB + seed data
    ├── Dockerfile
    └── requirements.txt
```

---

## Getting Started

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Docker + Docker Compose)
- ~4 GB disk space (Llama 3.2 model)
- No API key required — everything runs locally

### Run

```bash
# Clone the repository
git clone <repo-url>
cd simpleLanggraphMcp

# Start all services (first run downloads Llama 3.2 ~2 GB)
docker compose up --build -d

# Open the chat interface
open http://localhost:7860
```

> **First start:** the `ollama-init` service automatically pulls the Llama 3.2 model. This takes a few minutes once, then it's cached in a Docker volume.

### Services

| Service | URL | Description |
|---|---|---|
| Web UI | http://localhost:7860 | Chat interface |
| MCP Server | http://localhost:8080 | FastMCP SSE endpoint |
| ChromaDB | http://localhost:8001 | Vector database |
| Ollama | http://localhost:11434 | LLM runtime |

### Stop

```bash
docker compose down          # stop containers, keep data
docker compose down -v       # stop containers + delete knowledge base
```

---

## Adding Documents to the Knowledge Base

**Option 1 — Via the chat interface:**
```
Add this document to the knowledge base: "<your text here>" source: my_source
```

**Option 2 — In code** — edit the `documents` list in [`mcp_server/server.py`](mcp_server/server.py), then:
```bash
docker compose down -v
docker compose up --build -d
```

**Option 3 — Via the MCP API directly:**
```bash
# Check current document count
curl -N http://localhost:8080/sse
```

---

## Design Decisions

**Why MCP instead of direct function calls?**
MCP decouples the RAG capability from the agent. The same MCP server can be reused by any MCP-compatible client (Claude Desktop, other LangGraph agents, etc.) without code changes.

**Why Ollama + Llama 3.2 instead of a cloud LLM?**
Full local execution — no API costs, no data leaving the machine, works offline. The architecture is model-agnostic: swap `ChatOllama` for `ChatAnthropic` or `ChatOpenAI` with one line.

**Why SSE instead of WebSockets for streaming?**
SSE is unidirectional (server → client), simpler to implement, and natively supported by browsers without libraries. For a chat UI where the server streams responses, SSE is sufficient and more lightweight.

---

## Skills Demonstrated

- **Agentic AI systems** — multi-step reasoning with tool use (LangGraph ReAct)
- **Model Context Protocol** — building and consuming MCP servers
- **RAG pipeline** — document ingestion, vector embeddings, semantic retrieval
- **LLM integration** — local inference with Ollama, streaming responses
- **API design** — FastAPI, SSE, async Python
- **Containerisation** — Docker Compose, multi-stage builds, health checks, init containers
- **Production patterns** — buffered streaming, graceful startup, service dependencies

---

## License

MIT
