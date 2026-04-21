import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tavily import AsyncTavilyClient
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    function_tool,
    set_tracing_disabled,
)

load_dotenv()
set_tracing_disabled(True)

nvidia_client = AsyncOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],
)

qwen_model = OpenAIChatCompletionsModel(
    model="qwen/qwen3-coder-480b-a35b-instruct",
    openai_client=nvidia_client,
)

tavily_client = AsyncTavilyClient(api_key=os.environ["TAVILY_API_KEY"])


@function_tool
async def web_search(query: str, max_results: int = 5) -> str:
    """Search the live web and return a list of result snippets with URLs.

    Use this whenever the user asks about current events, recent news,
    specific facts, or anything that requires up-to-date information.

    Args:
        query: The search query. Keep it short and specific (3-8 words).
        max_results: How many results to return. Default 5, max 10.
    """
    print(f"[TOOL] web_search(query={query!r}, max_results={max_results})")
    try:
        response = await tavily_client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
        )
    except Exception as e:
        return f"Search failed: {e}"

    results = response.get("results", [])
    if not results:
        return "No results found."

    formatted = []
    for i, r in enumerate(results, start=1):
        formatted.append(
            f"[{i}] {r['title']}\n"
            f"    URL: {r['url']}\n"
            f"    {r['content'][:300]}"
        )
    return "\n\n".join(formatted)


@function_tool
async def fetch_page(url: str) -> str:
    """Fetch the full text content of a specific URL.

    Use this AFTER web_search when a snippet looks promising
    and you need the full article content to answer accurately.

    Args:
        url: The full URL to fetch, including https://.
    """
    print(f"[TOOL] fetch_page(url={url!r})")
    try:
        response = await tavily_client.extract(urls=[url])
    except Exception as e:
        return f"Fetch failed: {e}"

    results = response.get("results", [])
    if not results:
        return "Could not extract content from that URL."

    content = results[0].get("raw_content", "")
    return content[:4000] if content else "Page was empty."


research_agent = Agent(
    name="ResearchAssistant",
    instructions=(
        "You are a thorough research assistant. "
        "For any factual question: "
        "1. Call web_search first with a focused query. "
        "2. If a result looks highly relevant but the snippet is too short, "
        "   call fetch_page on that URL for full context. "
        "3. Synthesize a clear, concise answer citing the URLs you used. "
        "Always base your answer on search results, not your own knowledge."
    ),
    model=qwen_model,
    tools=[web_search, fetch_page],
)


async def main():
    question = "What are the most important AI developments in the last month?"
    print(f"QUESTION: {question}\n{'─' * 60}")

    result = await Runner.run(research_agent, input=question)

    print(f"\n{'═' * 60}")
    print("FINAL ANSWER:")
    print(f"{'═' * 60}")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
