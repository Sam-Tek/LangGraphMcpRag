"""
CLI entry point for the LangGraph + MCP + RAG agent.

Flow:
  User input → LangGraph ReAct loop → MCP tool call → ChromaDB search → LLM answer
"""

import asyncio
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx
from langchain_core.messages import HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# En Docker, __file__ est /app/main.py donc shared est /app/shared (parent/shared).
# En local, agent/main.py donc shared est ../../shared (parent.parent/shared).
_shared = Path(__file__).resolve().parent / "shared"
if not _shared.exists():
    _shared = _shared.parent.parent / "shared"
sys.path.insert(0, str(_shared))
from agent import MCP_SERVER_URL, HF_TOKEN, HF_MODEL_ID, managed_agent


def _wait_for_service(url: str, name: str, retries: int = 30, delay: float = 3.0) -> None:
    """Poll a service URL until it responds with a 2xx status code."""
    for attempt in range(retries):
        try:
            with httpx.Client(timeout=3) as client:
                response = client.get(url)
                response.raise_for_status()  # lève une exception si code >= 400
            print(f"{name} is ready.")
            return
        except Exception:
            print(f"Waiting for {name} ({attempt + 1}/{retries})...")
            time.sleep(delay)
    print(f"Warning: {name} may not be ready yet. Proceeding anyway...")


async def main() -> None:
    print("=" * 60)
    print(f"  LangGraph + MCP + RAG  —  Model: {HF_MODEL_ID}")
    print(f"  Provider   : HuggingFace Inference API")
    print(f"  MCP server : {MCP_SERVER_URL}")
    print("  Type 'quit' or press Ctrl-C to exit.")
    print("=" * 60)

    parsed = urlparse(MCP_SERVER_URL)
    mcp_base_url = f"{parsed.scheme}://{parsed.netloc}"
    _wait_for_service(f"{mcp_base_url}/health", "MCP server")

    # Modèle HuggingFace — initialisé une seule fois pour toute la session
    endpoint = HuggingFaceEndpoint(
        repo_id=HF_MODEL_ID,
        task="text-generation",
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=1024,
        temperature=0.1,
    )
    model = ChatHuggingFace(llm=endpoint)
    async with managed_agent(model) as agent:
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
                result = await agent.ainvoke(
                    {"messages": [HumanMessage(content=question)]}
                )
                # state["messages"] grows with each step:
                # [HumanMessage, AIMessage(tool_call), ToolMessage, AIMessage(final)]
                print(f"Answer:\n{result['messages'][-1].content}", flush=True)
            except Exception as exc:
                print(f"Error: {exc}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
