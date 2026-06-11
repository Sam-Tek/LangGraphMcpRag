"""
Shared agent logic used by both the CLI (agent/main.py) and the web service (web/app.py).

Centralising build_graph here avoids duplication and ensures both entry points
use exactly the same LangGraph wiring and system prompt.
"""

import os

from langchain_core.messages import SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

# ── Configuration ─────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MCP_SERVER_URL  = os.getenv("MCP_SERVER_URL",  "http://localhost:8080/sse")
MODEL_NAME      = os.getenv("OLLAMA_MODEL",    "llama3.2")

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a knowledge base. "
    "When a question requires specific information, always search the knowledge base first "
    "using the search_knowledge_base tool before answering. "
    "Cite the sources you used at the end of your answer."
)


def build_graph(model, tools):
    """
    Build the LangGraph ReAct graph (Reasoning + Acting loop).

    Graph structure:
        START → [agent] → (has tool calls?) → [tools] → [agent] → ...
                                           ↘ END

    Nodes:
        agent  — calls the LLM with the current message history.
        tools  — executes the MCP tool and returns the result to the agent.

    The loop continues until the LLM returns a message with no tool_calls,
    which signals it has enough information to answer.
    """
    # bind_tools tells the LLM which tools exist and what their schemas are
    model_with_tools = model.bind_tools(tools)
    # ToolNode routes tool_call requests to the correct MCP tool function
    tool_node = ToolNode(tools)

    def call_model(state: MessagesState):
        """Agent node: invoke the LLM with the full message history."""
        messages = state["messages"]
        # Inject the system prompt at position 0 if not already present
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


async def get_mcp_tools():
    """Connect to the MCP server and return the list of available tools."""
    mcp_client = MultiServerMCPClient(
        {"rag": {"url": MCP_SERVER_URL, "transport": "sse"}}
    )
    return await mcp_client.get_tools()
