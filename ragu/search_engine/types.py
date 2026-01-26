from dataclasses import dataclass, field
from textwrap import dedent

from jinja2 import Template


@dataclass
class LocalSearchResult:
    entities: list=field(default_factory=list)
    relations: list=field(default_factory=list)
    summaries: list=field(default_factory=list)
    chunks: list=field(default_factory=list)
    documents_id: list[str]=field(default_factory=list)

    _template: Template = Template(dedent(
        """
        **Entities**
        Entity, entity type, entity description
        {%- for e in entities %}
        {{ e.entity_name }}, {{ e.entity_type }}, {{ e.description }}
        {%- endfor %}
        
        **Relations**
        Subject, object, relation type, relation description, rank
        {%- for r in relations %}
        {{ r.subject_name }}, {{ r.object_name }}, {{ r.relation_type }} {{ r.description }}, {{ r.rank }}
        {%- endfor %}
        
        {%- if summaries %}
        **Summary**
        {%- for s in summaries %}
        {{ s }}
        {%- endfor %}
        {% endif %}
        
        {%- if chunks %}
        **Chunks**
        {%- for c in chunks %}
        {{ c.content }}
        {%- endfor %}
        {% endif %}
        """)
    )

    def __str__(self) -> str:
        return self._template.render(
            entities=self.entities,
            relations=self.relations,
            summaries=self.summaries,
            chunks=self.chunks,
        )


@dataclass
class GlobalSearchResult:
    insights: list=field(default_factory=list)

    _template: Template = Template(dedent(
        """
        {%- for insight in insights %}
        {{ loop.index}}. Insight: {{ insight.response }}, rating: {{ insight.rating }}
        {%- endfor %}
        """)
    )

    def __str__(self) -> str:
        return self._template.render(insights=self.insights)


@dataclass
class NaiveSearchResult:
    chunks: list=field(default_factory=list)
    scores: list=field(default_factory=list)
    documents_id: list[str]=field(default_factory=list)

    _template: Template = Template(dedent(
        """
        **Retrieved Chunks**
        {%- for chunk, score in zip(chunks, scores) %}
        [{{ loop.index }}] (score: {{ "%.3f"|format(score) }})
        {{ chunk.content }}
        {%- endfor %}
        """)
    )

    def __str__(self) -> str:
        return self._template.render(
            chunks=self.chunks,
            scores=self.scores,
            zip=zip,
        )
