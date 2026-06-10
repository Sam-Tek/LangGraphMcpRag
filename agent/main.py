"""
CLI entry point for the LangGraph + MCP + RAG agent.

Flow:
  User input → LangGraph ReAct loop → MCP tool call → ChromaDB search → LLM answer

The agent uses the ReAct pattern (Reasoning + Acting):
  1. The LLM receives the question and decides whether it needs more information.
  2. If yes, it emits a tool_call (e.g. search_knowledge_base).
  3. LangGraph intercepts the tool_call, runs it via the MCP server, and feeds
     the result back to the LLM as a tool_result message.
  4. The loop repeats until the LLM produces a final answer with no tool calls.
"""

import asyncio
import os
import sys
import time

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

# ── Configuration ────────────────────────────────────────────────────────────
# All values can be overridden via environment variables (see docker-compose.yml).
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MCP_SERVER_URL  = os.getenv("MCP_SERVER_URL",  "http://localhost:8080/sse")
MODEL_NAME      = os.getenv("OLLAMA_MODEL",    "llama3.2")

# The system prompt instructs the LLM to always search the knowledge base before
# answering. Without this, the LLM might answer from its training data alone.
SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a knowledge base. "
    "When a question requires specific information, always search the knowledge base first "
    "using the search_knowledge_base tool before answering. "
    "Cite the sources you used at the end of your answer."
)


def build_graph(model, tools):
    """
    Build a LangGraph StateGraph that implements the ReAct agentic loop.

    Graph structure:
        START → [agent] → (has tool calls?) → [tools] → [agent] → ...
                                           ↘ END

    Nodes:
        agent  — calls the LLM with the current message history.
        tools  — executes whatever tool the LLM requested (via MCP).

    The loop continues until the LLM returns a message with no tool_calls,
    which signals that it has enough information to answer.
    """
    # Bind the tools to the model so it knows what tools are available.
    # The LLM will include tool schemas in its context and can call them.
    model_with_tools = model.bind_tools(tools)

    # ToolNode automatically dispatches tool_call requests to the right function
    # and wraps the result in a ToolMessage for the LLM to read.
    tool_node = ToolNode(tools)

    def call_model(state: MessagesState):
        """Agent node: invoke the LLM with the full message history."""
        messages = state["messages"]
        # Inject the system prompt at position 0 if not already there.
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
        response = model_with_tools.invoke(messages)
        # Return a dict that LangGraph merges into the shared state.
        return {"messages": [response]}

    def should_continue(state: MessagesState):
        """
        Conditional edge: decide whether to call a tool or stop.

        If the last message contains tool_calls → route to "tools" node.
        Otherwise → route to END (the LLM has produced its final answer).
        """
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    # Build the graph
    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    # Conditional edge: after the agent node, check if we need to call tools
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    # After tools run, always go back to the agent for the next reasoning step
    graph.add_edge("tools", "agent")

    return graph.compile()


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
        2. Connect to the MCP server and fetch available tools.
           The MCP server exposes: search_knowledge_base, add_document, stats.
        3. Build the LangGraph ReAct graph with those tools.
        4. Invoke the graph; the loop runs until the LLM stops calling tools.
        5. Return the last message in the state (the final answer).
    """
    model = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)

    # MultiServerMCPClient connects to one or more MCP servers and converts
    # their tools into LangChain-compatible tool objects.
    mcp_client = MultiServerMCPClient(
        {"rag": {"url": MCP_SERVER_URL, "transport": "sse"}}
    )
    tools = await mcp_client.get_tools()

    agent = build_graph(model, tools)

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=question)]}
    )

    # The state["messages"] list grows with each step:
    # [HumanMessage, AIMessage(tool_call), ToolMessage, AIMessage(final)]
    # The last message is always the LLM's final answer.
    return result["messages"][-1].content


async def main() -> None:
    print("=" * 60)
    print(f"  LangGraph + MCP + RAG  —  Model: {MODEL_NAME}")
    print(f"  Ollama     : {OLLAMA_BASE_URL}")
    print(f"  MCP server : {MCP_SERVER_URL}")
    print("  Type 'quit' or press Ctrl-C to exit.")
    print("=" * 60)

    # Wait for dependencies to be ready before accepting input
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
