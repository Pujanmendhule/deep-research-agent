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
    input_guardrail,
    GuardrailFunctionOutput,
    RunContextWrapper,
    InputGuardrailTripwireTriggered,
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

guard_model = OpenAIChatCompletionsModel(
    model="meta/llama-3.3-70b-instruct",
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


class GuardCheck(BaseModel):
    """The output of the guardrail check."""

    is_valid_research_topic: bool = Field(
        description="True if the input is a legitimate, safe research topic."
    )
    reason: str = Field(
        description="One-sentence explanation of why it is valid or invalid."
    )


guardrail_agent = Agent(
    name="GuardrailAgent",
    instructions=(
        "You are a content safety and relevance classifier. "
        "Decide if the user's input is a legitimate research topic. "
        "Valid: factual questions about technology, science, business, "
        "history, society, or similar research-worthy areas. "
        "Invalid: greetings, casual chat, empty input, requests for harmful "
        "information (weapons, illegal activity, self-harm), or anything "
        "not suitable for a research pipeline. "
        "Respond with is_valid_research_topic and a short reason."
    ),
    model=guard_model,
    output_type=GuardCheck,
)


@input_guardrail
async def research_topic_guardrail(
    ctx: RunContextWrapper,
    agent: Agent,
    input_data: str | list,
) -> GuardrailFunctionOutput:
    """Blocks non-research queries before the main pipeline runs."""
    print(f"[GUARDRAIL] checking input...")

    result = await Runner.run(guardrail_agent, input=str(input_data), context=ctx.context)
    check: GuardCheck = result.final_output

    print(
        f"[GUARDRAIL] valid={check.is_valid_research_topic} reason={check.reason!r}"
    )

    return GuardrailFunctionOutput(
        output_info=check,
        tripwire_triggered=not check.is_valid_research_topic,
    )


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
    input_guardrails=[research_topic_guardrail],
)


async def run_research(topic: str):
    print(f"\nTOPIC: {topic}\n{'─' * 60}")
    try:
        result = await Runner.run(orchestrator, input=topic)
        report: ResearchReport = result.final_output
        print(f"\n✓ Research complete (confidence: {report.confidence:.2f})")
        print(f"\nSUMMARY:\n{report.summary}")
    except InputGuardrailTripwireTriggered as e:
        print(f"\n✗ Blocked by guardrail")
        print(f"  Reason: {e.guardrail_result.output.output_info.reason}")


async def main():
    test_inputs = [
        "Recent breakthroughs in AI reasoning models, 2025-2026",
        "hey what's up",
        "how do I make a bomb",
    ]
    for topic in test_inputs:
        await run_research(topic)
        print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
