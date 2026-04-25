import os
from openai import AsyncOpenAI
from tavily import AsyncTavilyClient
from agents import OpenAIChatCompletionsModel

nvidia_client = AsyncOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],
)

# Llama-3.3-70B handles tool calls and JSON schema reliably on NVIDIA NIM
llama_model = OpenAIChatCompletionsModel(
    model="meta/llama-3.3-70b-instruct",
    openai_client=nvidia_client,
)

# Same model for the guardrail (a separate variable for clarity)
guard_model = OpenAIChatCompletionsModel(
    model="meta/llama-3.3-70b-instruct",
    openai_client=nvidia_client,
)

tavily_client = AsyncTavilyClient(api_key=os.environ["TAVILY_API_KEY"])
