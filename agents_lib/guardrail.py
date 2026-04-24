from agents import (
    Agent,
    Runner,
    input_guardrail,
    GuardrailFunctionOutput,
    RunContextWrapper,
)
from agents_lib.models import guard_model
from agents_lib.schemas import GuardCheck


_guardrail_agent = Agent(
    name="GuardrailAgent",
    instructions=(
        "You are a content safety and relevance classifier. "
        "Decide if the user's input is a legitimate research topic. "
        "Valid: factual questions about technology, science, business, "
        "history, society, or similar research-worthy areas. "
        "Invalid: greetings, casual chat, empty input, requests for harmful "
        "information, or anything not suitable for a research pipeline. "
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
    result = await Runner.run(
        _guardrail_agent, input=str(input_data), context=ctx.context
    )
    check: GuardCheck = result.final_output

    return GuardrailFunctionOutput(
        output_info=check,
        tripwire_triggered=not check.is_valid_research_topic,
    )
