import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    function_tool,
    set_tracing_disabled,
)

load_dotenv()
set_tracing_disabled(True)

client = AsyncOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],
)

qwen_model = OpenAIChatCompletionsModel(
    model="qwen/qwen3-coder-480b-a35b-instruct",
    openai_client=client,
)


@function_tool
def fake_web_search(query: str) -> str:
    """Search the web for the given query and return relevant results.

    Args:
        query: The search query string.
    """
    print(f"[TOOL CALLED] fake_web_search(query={query!r})")
    fake_database = {
        "llm": "A Large Language Model is a neural network trained on massive text corpora to predict and generate text. Examples: GPT-4, Claude, Qwen.",
        "rag": "Retrieval-Augmented Generation combines a search step with an LLM so answers are grounded in retrieved documents rather than just training data.",
        "agent": "An AI agent is an LLM equipped with tools and a loop that lets it take multi-step actions toward a goal.",
    }
    for key, value in fake_database.items():
        if key in query.lower():
            return value
    return "No results found for that query."


agent = Agent(
    name="ResearchAssistant",
    instructions=(
        "You are a research assistant. When the user asks a factual question, "
        "use the fake_web_search tool to look up information before answering. "
        "Base your final answer on the search results."
    ),
    model=qwen_model,
    tools=[fake_web_search],
)

result = Runner.run_sync(agent, "What is RAG in AI?")
print("\n=== FINAL ANSWER ===")
print(result.final_output)
