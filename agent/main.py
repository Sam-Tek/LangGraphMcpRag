"""
CLI entry point for the LangGraph + MCP + RAG agent.

Flow:
  User input → LangGraph ReAct loop → MCP tool call → ChromaDB search → LLM answer
"""

import asyncio
import os
import sys
import time

import httpx
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

# Shared module — contains build_graph, get_mcp_tools, config constants
sys.path.insert(0, "/app/shared")
from agent import OLLAMA_BASE_URL, MCP_SERVER_URL, MODEL_NAME, build_graph, get_mcp_tools


def _wait_for_service(url: str, name: str, retries: int = 30, delay: float = 3.0) -> None:
    """Poll a service URL until it responds (startup synchronisation)."""
    for attempt in range(retries):
        try:
            with httpx.Client(timeout=3) as client:
                client.get(url)
            print(f"{name} is ready.")
            return
        except Exception:
            print(f"Waiting for {name} ({attempt + 1}/{retries})...")
            time.sleep(delay)
    print(f"Warning: {name} may not be ready yet. Proceeding anyway...")


async def ask(question: str) -> str:
    """
    Run a single question through the agentic loop and return the final answer.

    Steps:
        1. Instantiate the LLM (ChatOllama → local Llama via Ollama).
        2. Fetch available MCP tools from the RAG server.
        3. Build the LangGraph ReAct graph with those tools.
        4. Invoke the graph; the loop runs until the LLM stops calling tools.
        5. Return the last message in the state (the final answer).
    """
    model = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)
    tools = await get_mcp_tools()
    agent = build_graph(model, tools)

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=question)]}
    )

    # state["messages"] grows with each step:
    # [HumanMessage, AIMessage(tool_call), ToolMessage, AIMessage(final)]
    return result["messages"][-1].content


async def main() -> None:
    print("=" * 60)
    print(f"  LangGraph + MCP + RAG  —  Model: {MODEL_NAME}")
    print(f"  Ollama     : {OLLAMA_BASE_URL}")
    print(f"  MCP server : {MCP_SERVER_URL}")
    print("  Type 'quit' or press Ctrl-C to exit.")
    print("=" * 60)

    _wait_for_service(f"{OLLAMA_BASE_URL}/api/tags", "Ollama")
    _wait_for_service(MCP_SERVER_URL.replace("/sse", ""), "MCP server")

    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question or question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        print("\nThinking...\n", flush=True)
        try:
            answer = await ask(question)
            print(f"Answer:\n{answer}", flush=True)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
