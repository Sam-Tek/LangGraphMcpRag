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
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# Shared module — contains build_graph, get_mcp_tools, config constants
sys.path.insert(0, "/app/shared")
from agent import OLLAMA_BASE_URL, MODEL_NAME, build_graph, get_mcp_tools

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


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
    # streaming=True enables token-by-token generation in Ollama
    model = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, streaming=True)
    tools = await get_mcp_tools()
    agent = build_graph(model, tools)

    tool_called = False  # track whether we already notified the UI about RAG search
    word_buffer = ""     # accumulates sub-word tokens until a word boundary

    async for event in agent.astream_events(
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
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()


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
