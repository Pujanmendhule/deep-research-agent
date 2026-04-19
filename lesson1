import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled

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

agent = Agent(
    name="Assistant",
    instructions="You are a helpful research assistant. Answer clearly and concisely.",
    model=qwen_model,
)

result = Runner.run_sync(agent, "What is a Large Language Model, in one sentence?")
print(result.final_output)
