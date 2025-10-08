from dataclasses import asdict
from itertools import chain
from typing import List, Any

import pandas as pd

from ragu.common.base import RaguGenerativeModule
from ragu.common.prompts.default_models import RelationDescriptionModel, EntityDescriptionModel
from ragu.graph.types import Entity, Relation
from ragu.llm.base_llm import BaseLLM


class ArtifactsDescriptionSummarizer(RaguGenerativeModule):
    """
    Summarizes and merges textual descriptions of entities and relations
    extracted from documents.

    The class groups duplicated entities and relations by their identifiers
    (e.g., ``entity_name``, ``entity_type`` for entities, and
    ``subject_id``, ``object_id`` for relations), merges their attributes,
    and optionally generates concise descriptions through an LLM.

    :param client: LLM client used for summarization. Required if
                   ``use_llm_summarization=True``.
    :param use_llm_summarization: Whether to perform description summarization
                                  with a language model.
    :param language: Target language for summarization (e.g., ``"russian"`` or ``"english"``).
    :raises ValueError: If ``use_llm_summarization=True`` but no client is provided.
    """

    def __init__(self, client: BaseLLM = None, use_llm_summarization: bool = True, language: str = "russian"):
        _PROMPTS = ["entity_summarizer", "relation_summarizer"]
        super().__init__(prompts=_PROMPTS)

        self.client = client
        self.language = language
        self.use_llm_summarization = use_llm_summarization

        if self.use_llm_summarization and self.client is None:
            raise ValueError(
                "LLM summarization is enabled but no client is provided. Please provide a client."
            )

    async def run(self, entities: List[Entity], relations: List[Relation], **kwargs) -> Any:
        """
        Execute the full artifact summarization pipeline.

        The pipeline performs the following steps:

        1. Group duplicated entities and relations into aggregated dataframes.
        2. Summarize merged entity and relation descriptions if enabled.
        3. Return the updated lists of :class:`Entity` and :class:`Relation` objects.

        :param entities: List of extracted entities to summarize or merge.
        :param relations: List of extracted relations to summarize or merge.
        :return: A tuple ``(entities, relations)`` containing updated objects.
        """
        grouped_entities_df = self.group_entities(entities)
        grouped_relations_df = self.group_relations(relations)

        entities_to_return = await self.summarize_entities(grouped_entities_df)
        relations_to_return = await self.summarize_relations(grouped_relations_df)

        return entities_to_return, relations_to_return

    async def summarize_entities(self, grouped_entities_df: pd.DataFrame) -> List[Entity]:
        """
        Summarize merged entity descriptions.

        Entities with identical ``entity_name`` and ``entity_type`` are grouped
        into a single record. .

        :param grouped_entities_df: DataFrame containing grouped entity data with
                                    a ``duplicate_count`` column.
        :return: A list of summarized :class:`Entity` objects.
        """
        entity_mask = grouped_entities_df["duplicate_count"].to_numpy() > 1
        entity_multi_desc = grouped_entities_df.loc[entity_mask]
        entity_single_desc = grouped_entities_df.loc[~entity_mask]

        entity_multi_desc = entity_multi_desc.drop("duplicate_count", axis=1)
        entity_single_desc = entity_single_desc.drop("duplicate_count", axis=1)

        entities_to_summarize = []
        if len(entity_multi_desc) > 0 and self.use_llm_summarization:
            entities_to_summarize = [Entity(**row) for _, row in entity_multi_desc.iterrows()]
            prompt, schema = self.get_prompt("entity_summarizer").get_instruction(
                entity=entities_to_summarize,
                language=self.language,
            )
            response: List[EntityDescriptionModel] = await self.client.generate(prompt=prompt, schema=schema) # type: ignore

            for i, summary in enumerate(response):
                if summary:
                    entities_to_summarize[i].description = summary.description

        return [Entity(**row) for _, row in entity_single_desc.iterrows()] + entities_to_summarize

    async def summarize_relations(self, grouped_relations_df: pd.DataFrame) -> List[Relation]:
        """
        Summarize merged relation descriptions.

        Relations with identical pairs ``(subject_id, object_id)`` are combined
        into a single entry. If duplicates exist and LLM summarization is enabled,
        their descriptions are merged using the ``relation_summarizer`` prompt.

        :param grouped_relations_df: DataFrame containing grouped relation data with
                                     a ``duplicate_count`` column.
        :return: A list of summarized :class:`Relation` objects.
        """
        relation_mask = grouped_relations_df["duplicate_count"].to_numpy() > 1
        relation_multi_desc = grouped_relations_df.loc[relation_mask]
        relation_single_desc = grouped_relations_df.loc[~relation_mask]

        relation_multi_desc.drop("duplicate_count", axis=1, inplace=True)
        relation_single_desc.drop("duplicate_count", axis=1, inplace=True)

        relations_to_summarize = []
        if len(relation_multi_desc) > 0 and self.use_llm_summarization:
            relations_to_summarize = [Relation(**row) for _, row in relation_multi_desc.iterrows()]
            prompt, schema = self.get_prompt("relation_summarizer").get_instruction(
                relation=relations_to_summarize,
                language=self.language,
            )
            response: List[RelationDescriptionModel] = await self.client.generate(prompt=prompt, schema=schema) # type: ignore

            for i, summary in enumerate(response):
                if summary:
                    relations_to_summarize[i].description = summary.description

        return [Relation(**row) for _, row in relation_single_desc.iterrows()] + relations_to_summarize

    @staticmethod
    def group_entities(entities: List[Entity]) -> pd.DataFrame:
        """
        Group entities by ``entity_name`` and ``entity_type`` and aggregate their
        fields into combined records.

        :param entities: List of :class:`Entity` objects to group.
        :return: Aggregated entities as a :class:`pandas.DataFrame`.
        """
        entities_df = pd.DataFrame([asdict(entity) for entity in entities])
        grouped_entities = entities_df.groupby(["entity_name", "entity_type"]).agg(
            description=("description", lambda x: "\n".join(x.dropna().astype(str))),
            duplicate_count=("description", "count"),
            source_chunk_id=("source_chunk_id", lambda s: list(set(
                chain.from_iterable(v if isinstance(v, (list, tuple, set)) else [v] for v in s)))
            ),
            documents_id=("documents_id", lambda s: list(set(
                chain.from_iterable(v if isinstance(v, (list, tuple, set)) else [v] for v in s)))
            ),
        ).reset_index()

        return grouped_entities

    @staticmethod
    def group_relations(relations: List[Relation]) -> pd.DataFrame:
        """
        Group relations by ``subject_id`` and ``object_id`` and merge their fields.

        :param relations: List of :class:`Relation` objects to group.
        :return: Aggregated relations as a :class:`pandas.DataFrame`.
        """
        relations_df = pd.DataFrame([asdict(relation) for relation in relations])
        grouped_relations = relations_df.groupby(["subject_id", "object_id"]).agg(
            subject_name=("subject_name", "first"),
            object_name=("object_name", "first"),
            description=("description", lambda x: "\n".join(x.dropna().astype(str))),
            relation_strength=("relation_strength", "mean"),
            source_chunk_id=("source_chunk_id", lambda s: list(set(
                chain.from_iterable(v if isinstance(v, (list, tuple, set)) else [v] for v in s)))
            ),
            duplicate_count=("description", "count"),
        ).reset_index()

        return grouped_relations
