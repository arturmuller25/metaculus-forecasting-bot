"""
Claude web research through the Anthropic API, with the server-side web
search tool, for the "anthropic-search" research provider.

The same search engine OpenRouter uses for Claude's ":online" models, but paid
by the bot maker's own Anthropic credit instead of the tournament key: that
provider cost about $0.25 of the key per question (tokens plus search fees),
the largest single item after the GPT research. The reply keeps Claude's text
and appends the sources it cited, so the forecasters can see where each
claim came from.
"""

from __future__ import annotations

import os
from typing import Any

import anthropic

MAX_TOKENS = 8000
MAX_CONTINUATIONS = 3


def _blocks(message: Any) -> list[dict]:
    return [b.model_dump() if hasattr(b, "model_dump") else dict(b) for b in message.content]


async def research(prompt: str, model: str = "claude-sonnet-5", effort: str = "low",
                   max_searches: int = 6, timeout: float = 180.0) -> str:
    """Claude's research text followed by the list of cited sources."""
    if not os.getenv("ANTHROPIC_API_KEY", "").strip():
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    tools = [{"type": "web_search_20260318", "name": "web_search", "max_uses": max_searches}]
    messages: list[dict] = [{"role": "user", "content": prompt}]
    written: list[dict] = []
    async with anthropic.AsyncAnthropic(max_retries=1, timeout=timeout) as client:
        for _ in range(MAX_CONTINUATIONS + 1):
            async with client.messages.stream(
                model=model, max_tokens=MAX_TOKENS, messages=messages, tools=tools,
                thinking={"type": "adaptive"}, output_config={"effort": effort},
            ) as stream:
                msg = await stream.get_final_message()
            blocks = _blocks(msg)
            written.extend(blocks)
            if msg.stop_reason != "pause_turn":
                break
            # A long search turn pauses; resending continues it.
            if messages[-1]["role"] == "assistant":
                messages[-1]["content"].extend(blocks)
            else:
                messages.append({"role": "assistant", "content": list(blocks)})

    text = "".join(b.get("text", "") for b in written if b.get("type") == "text").strip()
    # Citations when the reply has them; with this tool version Claude often
    # filters results in code and cites nothing, so the pages it searched are
    # listed from the search results instead.
    cited: dict[str, str] = {}
    for b in written:
        for c in b.get("citations") or []:
            if c.get("url") and c["url"] not in cited:
                cited[c["url"]] = c.get("title") or ""
    searched: dict[str, str] = {}
    for b in written:
        if b.get("type") == "web_search_tool_result" and isinstance(b.get("content"), list):
            for item in b["content"]:
                url = item.get("url")
                if url and url not in searched and url not in cited:
                    searched[url] = " ".join(x for x in (item.get("title") or "", f"({item['page_age']})" if item.get("page_age") else "") if x)
    if not text:
        raise RuntimeError(f"no text in the reply (stop reason {msg.stop_reason})")
    if cited:
        text += "\n\nSources cited:\n" + "\n".join(f"- {t}: {u}" if t else f"- {u}" for u, t in cited.items())
    if searched:
        text += "\n\nPages found by the searches:\n" + "\n".join(f"- {t}: {u}" if t else f"- {u}" for u, t in list(searched.items())[:15])
    return text
