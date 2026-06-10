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

import asyncio
import os
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# ── Configuration ─────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MCP_SERVER_URL  = os.getenv("MCP_SERVER_URL",  "http://localhost:8080/sse")
MODEL_NAME      = os.getenv("OLLAMA_MODEL",    "llama3.2")

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a knowledge base. "
    "When a question requires specific information, always search the knowledge base first "
    "using the search_knowledge_base tool before answering. "
    "Cite the sources at the end of your answer."
)

app = FastAPI()
# Serve static files (index.html, CSS, JS) from the /static directory
app.mount("/static", StaticFiles(directory="static"), name="static")


def build_graph(model, tools):
    """
    Build the LangGraph ReAct graph (same pattern as agent/main.py).

    The graph alternates between two nodes:
        agent → decides what to do (call a tool or answer directly)
        tools → executes the MCP tool and returns the result to the agent
    """
    # bind_tools tells the LLM which tools exist and what their schemas are
    model_with_tools = model.bind_tools(tools)
    # ToolNode routes tool_call requests to the correct MCP tool function
    tool_node = ToolNode(tools)

    def call_model(state: MessagesState):
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: MessagesState):
        """Route to 'tools' if the LLM made a tool call, otherwise stop."""
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


async def stream_answer(question: str) -> AsyncIterator[str]:
    """
    Run the agentic loop and yield text chunks as they are generated.

    Uses agent.astream_events() which emits fine-grained events:
        on_tool_start       → a tool is about to be called
        on_chat_model_stream → the LLM is generating a token

    Word-boundary buffering:
        LLMs produce sub-word tokens (e.g. "é", "pre", "u", "ves" for "épreuves").
        We accumulate tokens in a buffer and only yield when we reach a space
        or newline, so the browser always receives complete words.
    """
    # streaming=True enables token-by-token generation in Ollama
    model = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, streaming=True)

    # Fetch the MCP tools (search_knowledge_base, add_document, stats)
    mcp_client = MultiServerMCPClient(
        {"rag": {"url": MCP_SERVER_URL, "transport": "sse"}}
    )
    tools = await mcp_client.get_tools()
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

        # Stream LLM output tokens (only from the final answer step,
        # not from the tool-calling step which also produces AI tokens)
        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                # Skip chunks that contain tool_call_chunks (those are
                # the LLM deciding WHICH tool to call, not the final answer)
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

    The browser opens a fetch() stream, reads the SSE lines, and appends
    each token to the chat bubble in real time. The special [DONE] sentinel
    tells the browser that the response is complete.
    """
    async def generator():
        async for token in stream_answer(req.question):
            yield {"data": token}
        yield {"data": "[DONE]"}  # sentinel: tells the browser to stop reading

    return EventSourceResponse(generator())


@app.get("/health")
async def health():
    """Health check used by Docker Compose to gate dependent services."""
    return {"status": "ok"}
