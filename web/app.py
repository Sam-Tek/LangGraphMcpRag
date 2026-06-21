"""
FastAPI web service — streaming chat interface for the LangGraph RAG agent.

Endpoints:
    GET  /       → serves the HTML chat UI (static/index.html)
    POST /ask    → runs the agentic loop and streams the answer via SSE
    GET  /health → health check used by Docker Compose

Streaming strategy:
    The LLM generates text token by token. We use Server-Sent Events (SSE)
    to push each token to the browser as it arrives. A word-boundary buffer
    prevents sub-word tokens from appearing split on screen.
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# En Docker, __file__ est /app/app.py donc shared est /app/shared (parent/shared).
# En local, web/app.py donc shared est ../../shared (parent.parent/shared).
_shared = Path(__file__).resolve().parent / "shared"
if not _shared.exists():
    _shared = _shared.parent.parent / "shared"
sys.path.insert(0, str(_shared))
from agent import HF_TOKEN, HF_MODEL_ID, managed_agent

STATIC_DIR = Path(__file__).parent / "static"

# Agent partagé entre toutes les requêtes, initialisé au démarrage de l'app
_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise le modèle HuggingFace et l'agent une seule fois; ferme la connexion MCP à l'arrêt."""
    global _agent
    endpoint = HuggingFaceEndpoint(
        repo_id=HF_MODEL_ID,
        task="text-generation",
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=1024,
        temperature=0.1,
    )
    model = ChatHuggingFace(llm=endpoint)
    async with managed_agent(model) as agent:
        _agent = agent
        yield
    _agent = None


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


async def stream_answer(question: str) -> AsyncIterator[str]:
    """
    Run the agentic loop and yield text chunks as they are generated.

    Uses agent.astream_events() which emits fine-grained events:
        on_tool_start        → a tool is about to be called
        on_chat_model_stream → the LLM is generating a token

    Word-boundary buffering:
        LLMs produce sub-word tokens (e.g. "é", "pre", "u", "ves").
        We accumulate tokens in a buffer and only yield when we reach a space
        or newline, so the browser always receives complete words.
    """
    tool_called = False  # track whether we already notified the UI about RAG search
    word_buffer = ""     # accumulates sub-word tokens until a word boundary

    async for event in _agent.astream_events(
        {"messages": [HumanMessage(content=question)]},
        version="v2",  # v2 is required for on_chat_model_stream events
    ):
        kind = event["event"]

        # Notify the browser the first time a tool is invoked
        if kind == "on_tool_start" and not tool_called:
            tool_called = True
            yield "🔍 Searching knowledge base...\n\n"

        # Stream LLM output tokens (only from the final answer step)
        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                # Skip tool_call_chunks — those are the LLM deciding WHICH tool
                # to call, not the final answer tokens
                if not (hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks):
                    word_buffer += chunk.content
                    # Flush to the browser at each word boundary
                    if " " in word_buffer or "\n" in word_buffer:
                        last_space = max(
                            word_buffer.rfind(" "),
                            word_buffer.rfind("\n"),
                        )
                        yield word_buffer[: last_space + 1]
                        word_buffer = word_buffer[last_space + 1:]

    # Flush any remaining text that didn't end with a space
    if word_buffer:
        yield word_buffer


class QuestionRequest(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the chat UI."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=503, detail="UI not available")
    return index_file.read_text(encoding="utf-8")


@app.post("/ask")
async def ask(req: QuestionRequest):
    """
    Streaming endpoint — runs the agent and pushes tokens via SSE.

    The [DONE] sentinel tells the browser the response is complete.
    """
    async def generator():
        async for token in stream_answer(req.question):
            yield {"data": token}
        yield {"data": "[DONE]"}

    return EventSourceResponse(generator())


@app.get("/health")
async def health():
    """Health check used by Docker Compose to gate dependent services."""
    return {"status": "ok"}
