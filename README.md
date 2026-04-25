# 🔍 Deep Research Agent

A multi-agent deep research system built with the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python).
Give it a topic and it returns a structured research report — backed by real web sources, validated by safety guardrails, and produced through coordinated specialist agents.

```bash
python research.py "recent breakthroughs in AI reasoning models"
```

```
14:03:02  starting research: 'recent breakthroughs in AI reasoning models'
14:03:14  research complete (confidence 0.80)
14:03:14  report saved: reports/20260425-140314-recent-breakthroughs.md
```

---

## What it does

- 🛡️ **Guardrails** the input — rejects greetings, harmful queries, and off-topic requests before any expensive work runs
- 🔎 **Searches the live web** through Tavily, gathering 10–15 sources across multiple angle queries
- 🤝 **Hands off** between specialist agents — a search agent and a senior analyst — each with focused instructions
- 📋 **Returns structured output** — a Pydantic-validated report with summary, findings, open questions, sources, and a confidence score
- 💾 **Saves a markdown report** to `reports/` for every run

## Architecture

```
                    User topic
                         │
                  ┌──────▼──────┐
                  │  Guardrail  │  ← Llama-3.3-70B classifier
                  │  (safety)   │     blocks bad inputs
                  └──────┬──────┘
                         │ if safe
                  ┌──────▼──────┐
                  │Orchestrator │  ← routes to search agent
                  └──────┬──────┘
                         │
                  ┌──────▼──────┐
                  │   Search    │  ← Llama-3.3-70B + web_search tool
                  │    agent    │     calls Tavily 2-3 times
                  └──────┬──────┘
                         │ handoff with raw findings
                  ┌──────▼──────┐
                  │   Analyst   │  ← Llama-3.3-70B + Pydantic schema
                  │    agent    │     produces typed ResearchReport
                  └──────┬──────┘
                         │
                ┌────────▼────────┐
                │ markdown report │
                │ saved to disk   │
                └─────────────────┘
```

## Why this design

| Design choice | Reason |
|---------------|--------|
| Multi-agent over monolith | Each agent has tight, focused instructions instead of one sprawling prompt |
| Guardrail uses a separate classifier | Cheap, fast safety check on every input — never wastes API calls on bad queries |
| Pydantic structured output | Downstream code can rely on `report.summary`, `report.confidence` instead of parsing text |
| NVIDIA NIM as provider | OpenAI-compatible endpoint with free open-weight models (Llama, Qwen, etc.) |

## Tech stack

- **[OpenAI Agents SDK](https://github.com/openai/openai-agents-python)** — agent orchestration and handoffs
- **[Pydantic](https://docs.pydantic.dev)** — schema-driven structured outputs
- **[NVIDIA NIM](https://build.nvidia.com)** — free hosted open-weight LLMs (Llama-3.3-70B)
- **[Tavily](https://tavily.com)** — LLM-optimized web search API
- **Python 3.10+** with async/await throughout

## Setup

### 1. Clone and install

```bash
git clone https://github.com/Pujan02/deep-research-agent.git
cd deep-research-agent

python -m venv .venv
.venv\Scripts\activate              # Windows
# source .venv/bin/activate         # macOS/Linux

pip install -r requirements.txt
```

### 2. Get free API keys

- **NVIDIA NIM** → [build.nvidia.com](https://build.nvidia.com) — sign in, copy any model's API key (starts with `nvapi-`)
- **Tavily** → [tavily.com](https://tavily.com) — 1,000 free searches/month, no credit card

### 3. Create a `.env` file in the project root

```env
NVIDIA_API_KEY=nvapi-your-key-here
TAVILY_API_KEY=tvly-your-key-here
```

### 4. Run it

```bash
python research.py "your research topic in quotes"
```

The report is saved to `reports/` as a markdown file.

## Project structure

```
deep-research-agent/
├── research.py              # CLI entry point
├── agents_lib/              # the agent system
│   ├── __init__.py
│   ├── models.py            # NVIDIA + Tavily clients, model objects
│   ├── tools.py             # @function_tool web_search
│   ├── schemas.py           # Pydantic: GuardCheck, ResearchReport
│   ├── guardrail.py         # @input_guardrail safety check
│   └── pipeline.py          # the 3 research agents
├── reports/                 # generated reports (gitignored)
├── lesson1.py … lesson6.py  # tutorial scripts (incremental learning)
├── requirements.txt
├── .env                     # gitignored — your API keys
└── .gitignore
```

## How it works under the hood

### The agent loop

Every agent runs the same loop: send input + history to the LLM → parse the response → if it's a tool call, run the tool and feed the result back → if it's a handoff, transfer control to the target agent → if it's a final answer, return.

The SDK handles this loop for you. You just configure agents and let `Runner.run()` orchestrate.

### Handoffs vs. tools

- A **tool** is a Python function the agent calls and gets a result from
- A **handoff** transfers control entirely — the receiving agent takes over and its output becomes the run's output

This project uses both: `web_search` is a tool, `search_agent → analyst_agent` is a handoff.

### Why the same model is used for all three agents

In testing, smaller open-weight models (Qwen-Coder, Llama-4-Maverick) on NVIDIA NIM had inconsistent tool-calling behavior under handoff chains. Llama-3.3-70B reliably calls tools, executes handoffs, and produces valid Pydantic output — so it's used everywhere for stability.

This is a pragmatic production trade-off: the right model for the job in *theory* (small for guardrails, large for analysis) loses to the model that *actually works reliably* across all roles.

## Example output

```markdown
# Research report: how do quantum computers solve cryptography problems
*Generated 2026-04-25 14:03:14 — confidence 0.80*

## Summary
Quantum computers can solve cryptography problems by using quantum
algorithms that can efficiently factor large numbers and solve discrete
logarithms — the basis for many current encryption methods...

## Key findings
- Quantum computers can break many current encryption methods
- Post-quantum cryptography algorithms are being developed to address this threat
- The US government has established guidelines for post-quantum cryptography

## Open questions
- How soon will quantum computers be able to break current encryption methods?
- How widely will post-quantum cryptography be adopted?

## Sources
- https://www.livescience.com/technology/computing/...
- https://csrc.nist.gov/projects/post-quantum-cryptography
- https://www.cloudflare.com/learning/ssl/quantum/...
```

## Roadmap / ideas to extend

- [ ] **Output guardrail** — auto-reject reports below confidence 0.5 and retry with a different angle
- [ ] **Sessions** — multi-turn refinement of a single report ("dig deeper into finding 3")
- [ ] **PDF export** — generate a polished PDF in addition to markdown
- [ ] **FastAPI wrapper** — turn the CLI into a web service
- [ ] **Domain-specific clones** — same architecture for sales intel, legal contract review, due diligence

## What I learned building this

This was built as a hands-on way to learn the OpenAI Agents SDK from scratch. Key takeaways:

1. **An LLM is just text-in, text-out.** Tools are what give it real capability.
2. **The agent loop is the framework's whole reason to exist.** It saves you from writing the "call model → parse tool call → run tool → feed back → call model" glue every time.
3. **Provider quirks are real.** "OpenAI-compatible" endpoints don't all behave identically — tool-calling and structured output are the two fragile primitives.
4. **Multi-agent isn't always worth it.** It's powerful when each agent has a distinctly different job; otherwise a single agent with multiple tools is simpler.
5. **Pydantic + `output_type` is the killer feature.** Once you have typed structured output, downstream code becomes trivial.

## License

MIT
