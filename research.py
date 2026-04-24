import sys
import asyncio
import logging
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from agents import Runner, InputGuardrailTripwireTriggered, set_tracing_disabled
from agents_lib.pipeline import orchestrator
from agents_lib.schemas import ResearchReport

set_tracing_disabled(True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("research")


REPORTS_DIR = Path("reports")


def format_report_as_markdown(topic: str, report: ResearchReport) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"# Research report: {topic}",
        "",
        f"*Generated {timestamp} — confidence {report.confidence:.2f}*",
        "",
        "## Summary",
        report.summary,
        "",
        "## Key findings",
    ]
    for finding in report.key_findings:
        lines.append(f"- {finding}")

    lines.append("")
    lines.append("## Open questions")
    for q in report.open_questions:
        lines.append(f"- {q}")

    lines.append("")
    lines.append("## Sources")
    for s in report.sources:
        lines.append(f"- {s}")

    return "\n".join(lines)


def save_report(topic: str, markdown: str) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    slug = "".join(c if c.isalnum() else "-" for c in topic.lower())[:60].strip("-")
    filename = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{slug}.md"
    path = REPORTS_DIR / filename
    path.write_text(markdown, encoding="utf-8")
    return path


async def research(topic: str) -> int:
    log.info(f"starting research: {topic!r}")

    try:
        result = await Runner.run(orchestrator, input=topic)
    except InputGuardrailTripwireTriggered as e:
        reason = e.guardrail_result.output.output_info.reason
        log.warning(f"blocked by guardrail: {reason}")
        return 1
    except Exception as e:
        log.error(f"pipeline failed: {e}")
        return 2

    report: ResearchReport = result.final_output
    log.info(f"research complete (confidence {report.confidence:.2f})")

    markdown = format_report_as_markdown(topic, report)
    path = save_report(topic, markdown)
    log.info(f"report saved: {path}")

    return 0


def main():
    if len(sys.argv) < 2:
        print("Usage: python research.py \"your research topic\"")
        sys.exit(1)

    topic = " ".join(sys.argv[1:])
    exit_code = asyncio.run(research(topic))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
