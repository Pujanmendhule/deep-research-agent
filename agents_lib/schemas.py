from pydantic import BaseModel, Field


class GuardCheck(BaseModel):
    """Output of the input guardrail."""

    is_valid_research_topic: bool = Field(
        description="True if the input is a legitimate, safe research topic."
    )
    reason: str = Field(
        description="One-sentence explanation of why it is valid or invalid."
    )


class ResearchReport(BaseModel):
    """Final structured output of a deep research run."""

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
