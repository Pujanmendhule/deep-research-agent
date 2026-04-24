import os
from openai import AsyncOpenAI
from tavily import AsyncTavilyClient
from agents import OpenAIChatCompletionsModel

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
