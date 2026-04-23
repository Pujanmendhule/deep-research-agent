import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tavily import AsyncTavilyClient
from pydantic import BaseModel, Field
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

maverick_model = OpenAIChatCompletionsModel(
    model="meta/llama-4-maverick-17b-128e-instruct",
    openai_client=nvidia_client,
)

tavily_client = AsyncTavilyClient(api_key=os.environ["TAVILY_API_KEY"])


@function_tool
async def web_search(query: str, max_results: int = 5) -> str:
    """Search the live web and return a list of result snippets with URLs.

    Args:
        query: The search query. Keep it short and specific.
        max_results: How many results to return. Default 5, max 10.
    """
    print(f"[TOOL] web_search(query={query!r})")
    try:
        response = await tavily_client.search(
            query=query, max_results=max_results, search_depth="basic"
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


class ResearchReport(BaseModel):
    """The structured output of a deep research run."""

    summary: str = Field(description="A 2-3 sentence executive summary of the topic.")
    key_findings: list[str] = Field(
        description="5-7 factual findings, each a complete sentence."
    )
    open_questions: list[str] = Field(
        description="2-4 questions that remain unanswered or need more research."
    )
    sources: list[str] = Field(description="URLs used as sources for the report.")
    confidence: float = Field(
        description="Confidence in the report's accuracy, from 0.0 to 1.0."
    )


analyst_agent = Agent(
    name="AnalystAgent",
    handoff_description="Specialist that reads raw research notes and writes a structured report.",
    instructions=(
        "You are a senior research analyst. "
        "Given raw research notes with URLs, produce a ResearchReport. "
        "Every field must be filled based ONLY on the provided notes. "
        "Set 'confidence' lower if the notes are thin, conflicting, or outdated."
    ),
    model=qwen_model,
    output_type=ResearchReport,
)


search_agent = Agent(
    name="SearchAgent",
    handoff_description="Specialist that searches the web and gathers raw sources on a topic.",
    instructions=(
        "You are a web research specialist. "
        "When given a topic, call web_search 2-3 times with different angle queries. "
        "Compile ALL raw findings as a bulleted list, each with its URL. "
        "Once findings are complete, hand off to AnalystAgent to write the structured report. "
        "Do NOT write the final report yourself."
    ),
    model=maverick_model,
    tools=[web_search],
    handoffs=[analyst_agent],
)


orchestrator = Agent(
    name="ResearchOrchestrator",
    instructions=(
        "You coordinate a deep research pipeline. "
        "Given any topic, immediately hand off to SearchAgent. "
        "Do not do any research or writing yourself."
    ),
    model=qwen_model,
    handoffs=[search_agent],
)


async def main():
    topic = "Recent breakthroughs in AI reasoning models, 2025-2026"
    print(f"TOPIC: {topic}\n{'─' * 60}")

    result = await Runner.run(orchestrator, input=topic)

    report: ResearchReport = result.final_output

    print(f"\n{'═' * 60}")
    print("STRUCTURED REPORT:")
    print(f"{'═' * 60}")
    print(f"\nSUMMARY:\n{report.summary}")
    print(f"\nKEY FINDINGS:")
    for i, finding in enumerate(report.key_findings, 1):
        print(f"  {i}. {finding}")
    print(f"\nOPEN QUESTIONS:")
    for i, q in enumerate(report.open_questions, 1):
        print(f"  {i}. {q}")
    print(f"\nSOURCES:")
    for s in report.sources:
        print(f"  - {s}")
    print(f"\nCONFIDENCE: {report.confidence:.2f}")

    print(f"\n{'─' * 60}")
    print("RAW JSON:")
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
