from dataclasses import dataclass, field

from jinja2 import Template


@dataclass
class LocalSearchResult:
    entities: list=field(default_factory=list)
    relations: list=field(default_factory=list)
    summaries: list=field(default_factory=list)
    chunks: list=field(default_factory=list)

    _template: Template = Template(
"""
**Entities**\nEntity, entity type, entity description
{%- for e in entities %}
{{ e.entity_name }}, {{ e.entity_type }}, {{ e.description }}
{%- endfor %}

**Relations**\nSubject, object, relation description, rank
{%- for r in relations %}
{{ r.subject_name }}, {{ r.object_name }}, {{ r.description }}, {{ r.rank }}
{%- endfor %}

{%- if summaries %}
Summary
{%- for s in summaries %}
{{ s }}
{%- endfor %}
{% endif %}

{%- if chunks %}
Chunks
{%- for c in chunks %}
{{ c.content }}
{%- endfor %}
{% endif %}
"""
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

    _template: Template = Template(
        """
        {%- for insight in insights %} 
        {{ loop.index}}. Insight: {{ insight.response }}, rating: {{ insight.rating }}
        {%- endfor %}
        """.strip()
    )

    def __str__(self) -> str:
        return self._template.render(insights=self.insights)
