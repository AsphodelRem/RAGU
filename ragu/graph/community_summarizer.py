from textwrap import dedent
from typing import List
from jinja2 import Template

from ragu.common.base import RaguGenerativeModule
from ragu.common.global_parameters import Settings
from ragu.common.prompts.default_models import CommunityReportModel
from ragu.common.prompts.prompt_storage import RAGUInstruction
from ragu.common.prompts.messages import ChatMessages, render
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
        LLM client used for generating community reports.
    language : str
        Language of generated summaries.
    """

    def __init__(self, client: BaseLLM, language: str | None = None) -> None:
        _PROMPTS = ["community_report"]
        super().__init__(prompts=_PROMPTS)

        self.client = client
        self.language = language if language else Settings.language

    async def summarize(self, communities: List[Community]) -> List[CommunitySummary]:
        """
        Generate structured summaries for a list of graph communities.
        """
        sorted_communities = []
        for community in communities:
            sorted_communities.append(
                Community(entities=sorted(
                    community.entities, key=lambda e: e.id),
                    relations=sorted(community.relations, key=lambda e: e.id),
                    level=community.level,
                    cluster_id=community.cluster_id,
                )
            )
        instruction: RAGUInstruction = self.get_prompt("community_report")

        rendered_list: List[ChatMessages] = render(
            instruction.messages,
            community=sorted_communities,
            language=self.language,
        )

        summaries: List[CommunityReportModel] = await self.client.generate(  # type: ignore
            conversations=rendered_list,
            response_model=instruction.pydantic_model,
            progress_bar_desc="Summarized communities",
        )

        output: List[CommunitySummary] = [
            CommunitySummary(
                id=community.id,
                summary=self.combine_report_text(summary),
            )
            for (community, summary) in zip(sorted_communities, summaries)
        ]

        return output

    @staticmethod
    def combine_report_text(report: CommunityReportModel) -> str:
        """
        Merge structured sections of a community report into a readable text block.
        """
        if not report:
            return ""

        template = Template(dedent(
            """
            Report title: {{ report.title }}
            Report summary: {{ report.summary }}
            
            {% for finding in report.findings %}
            Finding summary: {{ finding.summary }}
            Finding explanation: {{ finding.explanation }}
            {% endfor %}
            """)
        )

        return template.render(report=report)
