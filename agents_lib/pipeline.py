from agents import Agent
from agents_lib.models import llama_model
from agents_lib.tools import web_search
from agents_lib.schemas import ResearchReport
from agents_lib.guardrail import research_topic_guardrail


analyst_agent = Agent(
    name="AnalystAgent",
    handoff_description="Specialist that reads raw research notes and writes a structured report.",
    instructions=(
        "You are a senior research analyst. "
        "Given raw research notes with URLs, produce a ResearchReport. "
        "Every field must be filled based ONLY on the provided notes. "
        "Set 'confidence' lower if the notes are thin, conflicting, or outdated."
    ),
    model=llama_model,
    output_type=ResearchReport,
)


search_agent = Agent(
    name="SearchAgent",
    handoff_description="Specialist that searches the web and gathers raw sources on a topic.",
    instructions=(
        "You are a web research specialist. "
        "When given a topic, call the web_search tool 2-3 times with different angle queries "
        "to gather diverse sources. "
        "Compile ALL raw findings as a bulleted list, each with its URL. "
        "Once findings are complete, hand off to AnalystAgent to write the structured report. "
        "Do NOT write the final report yourself."
    ),
    model=llama_model,
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
    model=llama_model,
    handoffs=[search_agent],
    input_guardrails=[research_topic_guardrail],
)
