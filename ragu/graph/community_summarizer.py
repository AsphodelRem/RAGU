from typing import List
from jinja2 import Template

from ragu.common.base import RaguGenerativeModule
from ragu.common.prompts.default_models import CommunityReportModel
from ragu.graph.types import Community, CommunitySummary
from ragu.llm.base_llm import BaseLLM


class CommunitySummarizer(RaguGenerativeModule):
    """
    Generates textual summaries for detected graph communities using an LLM.

    The summarization process typically converts a group of entities or
    relations belonging to the same community into a human-readable report
    containing a title, overall summary, and list of findings.

    Attributes
    ----------
    client : BaseLLM
        The underlying LLM client used for generating community summaries.
    language : str
        Language of generated summaries.
    """

    def __init__(self, client: BaseLLM, language: str = "english") -> None:
        _PROMPT = ["community_report"]
        super().__init__(prompts=_PROMPT)
        self.client = client
        self.language = language

    async def summarize(self, communities: List[Community]) -> List[CommunitySummary]:
        """
        Generate structured summaries for a list of graph communities.
        """
        instructions, schema = self.get_prompt("community_report").get_instruction(
            community=communities,
            language=self.language,
        )

        summaries: List[CommunityReportModel] = await self.client.generate(  # type: ignore
            prompt=instructions,
            schema=schema,
        )

        output: List[CommunitySummary] = [
            CommunitySummary(
                id=community.id,
                summary=self.combine_report_text(summary),
            )
            for (community, summary) in zip(communities, summaries)
        ]

        return output

    @staticmethod
    def combine_report_text(report: CommunityReportModel) -> str:
        """
        Merge structured sections of a community report into a readable text block.
        """
        if not report:
            return ""

        template = Template(
            """
            Report title: {{ report.title }}
            Report summary: {{ report.summary }}

            {% for finding in report.findings %}
            Finding summary: {{ finding.summary }}
            Finding explanation: {{ finding.explanation }}
            {% endfor %}
            """.strip()
        )

        return template.render(report=report)
