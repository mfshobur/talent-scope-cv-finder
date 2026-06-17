"""
Direct LiteLLM tool-calling loop — no framework.
run_agent() yields response text and returns updated message history.
"""

import json
from collections.abc import Generator

import litellm

litellm.drop_params = True

from agent.prompts import SYSTEM_PROMPT
from agent.tools import build_tool_schemas, dispatch_tool
from config.settings import Settings
from ingestion.chromadb_store import CandidateStore
from core.embedder import Embedder


def run_agent(
    query: str,
    history: list[dict],
    store: CandidateStore,
    embedder: Embedder,
    settings: Settings,
) -> Generator[str, None, list[dict]]:
    """
    Yields response text chunks.
    After the generator is exhausted, the caller should read the returned history
    via the StopIteration value or the last yielded state.

    Usage:
        gen = run_agent(query, history, store, embedder, settings)
        full_response = ""
        updated_history = history
        try:
            while True:
                chunk = next(gen)
                full_response += chunk
        except StopIteration as e:
            updated_history = e.value
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [
        {"role": "user", "content": query}
    ]
    tools = build_tool_schemas()
    api_key = settings.effective_llm_api_key() or None
    base_url = settings.llm_base_url or None

    while True:
        response = litellm.completion(
            model=settings.llm_model,
            messages=messages,
            tools=tools,
            api_key=api_key,
            base_url=base_url,
            temperature=0.3,
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            # Execute tools and append results to messages
            messages.append(msg.model_dump(exclude_none=True))
            for call in msg.tool_calls:
                result = dispatch_tool(
                    call.function.name,
                    call.function.arguments,
                    store,
                    embedder,
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })
        else:
            # Final text response
            content = msg.content or ""
            yield content
            messages.append({"role": "assistant", "content": content})
            return messages[1:]  # strip system prompt — caller prepends it each turn


def collect_response(
    query: str,
    history: list[dict],
    store: CandidateStore,
    embedder: Embedder,
    settings: Settings,
) -> tuple[str, list[dict]]:
    """Convenience wrapper — collects full response and returns (text, updated_history)."""
    gen = run_agent(query, history, store, embedder, settings)
    full_text = ""
    updated_history = history
    try:
        while True:
            full_text += next(gen)
    except StopIteration as e:
        if e.value is not None:
            updated_history = e.value
    return full_text, updated_history
