from typing import List, Tuple

import pandas as pd
from tqdm import tqdm

from ragu.common import GlobalPromptStorage
from ragu.common.batch_generator import BatchGenerator
from ragu.common.llm import BaseLLM
from ragu.triplet.base_triplet import TripletExtractor


@TripletExtractor.register("original")
class TripletLLM(TripletExtractor):
    """
    Extracts entities and relationships from text using LLM with absolute chunk indexing.
    """
    ENTITY_COLUMNS = ["entity_name", "entity_type", "description", "chunk_id"]
    RELATION_COLUMNS = ["source_entity", "target_entity", "relationship_description", "relationship_strength",
                        "chunk_id"]

    def __init__(
            self,
            batch_size: int = 16,
            validate: bool = False,
    ):
        """
        Initializes the TripletLLM extractor.

        :param validate: Flag to enable triplet validation
        :param batch_size: Number of texts to process per batch
        """

        super().__init__()
        self.validate = validate
        self.batch_size = batch_size

        self.artifact_extractor_prompt_tool = GlobalPromptStorage.artifact_extractor_prompt
        self.artifact_validation_prompt_tool = GlobalPromptStorage.artifacts_validation_prompt

    def extract_entities_and_relationships(
            self,
            chunks_df: pd.DataFrame,
            client: BaseLLM = None,
            *args,
            **kwargs
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Processes text in batches while preserving original corpus indices.

        :param chunks_df: DataFrame containing text chunks after splitting by chunker
        :param client: LLM client instance
        :return: Tuple containing two DataFrames:
            - Entities DataFrame with columns: entity_name, entity_type, chunk_id
            - Relationships DataFrame with columns: source_entity, target_entity,
              relationship_description, chunk_id
        """

        entities, relations = [], []
        text_df = chunks_df["chunk"]
        chunks_id_df = chunks_df["chunk_id"]

        batch_generator = BatchGenerator(
            list(zip(text_df.tolist(), chunks_id_df.tolist())),
            batch_size=self.batch_size
        )

        for batch_idx, batch in tqdm(
                enumerate(batch_generator.get_batches()),
                desc="Index creation: extracting entities and relationships",
                total=len(batch_generator)
        ):
            texts = [row[0] for row in batch]
            chunks_id = [row[1] for row in batch]

            batched_inputs = [{"context": text} for text in texts]
            parsed_batch = self.artifact_extractor_prompt_tool.batch_forward(client, batched_inputs)

            if self.validate:
                validation_inputs = [
                    {"text": text, "triplets": parsed_data}
                    for text, parsed_data in zip(texts, parsed_batch)
                ]
                parsed_batch = self.artifact_validation_prompt_tool.batch_forward(client, validation_inputs)

            self._process_parsed_batch(parsed_batch, chunks_id, entities, relations)

        return self._finalize_dataframes(entities, relations)

    def _process_parsed_batch(
            self,
            parsed_batch: List[dict],
            chunks_id: list,
            entities: List[pd.DataFrame],
            relations: List[pd.DataFrame]
    ) -> None:
        """
        Processes parsed batch data with absolute indexing.

        :param parsed_batch: Parsed data from current batch (dict with 'entities' and 'relationships')
        :param chunks_id: List of chunk IDs for the batch
        :param entities: List to accumulate entity DataFrames
        :param relations: List to accumulate relationship DataFrames
        """
        for i, data in enumerate(parsed_batch):
            entity_df = pd.DataFrame(data["entities"], columns=self.ENTITY_COLUMNS[:-1])
            relation_df = pd.DataFrame(data["relationships"], columns=self.RELATION_COLUMNS[:-1])

            entity_df["chunk_id"] = chunks_id[i]
            relation_df["chunk_id"] = chunks_id[i]

            entities.append(entity_df)
            relations.append(relation_df)

    def _finalize_dataframes(
            self,
            entities: List[pd.DataFrame],
            relations: List[pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Creates final output DataFrames from accumulated results.

        :param entities: List of entity DataFrames
        :param relations: List of relationship DataFrames
        :return: Tuple of concatenated DataFrames
        """
        entities = pd.concat(entities, ignore_index=True) if entities else \
            pd.DataFrame(columns=self.ENTITY_COLUMNS)
        relations = pd.concat(relations, ignore_index=True) if relations else \
            pd.DataFrame(columns=self.RELATION_COLUMNS)

        entities.dropna(inplace=True)
        relations.dropna(inplace=True)

        # Removing "ё" is additional normalization for Russian language
        entities["entity_name"] = entities["entity_name"].str.upper().str.replace('Ё', 'Е')
        relations["source_entity"] = relations["source_entity"].str.upper().str.replace('Ё', 'Е')
        relations["target_entity"] = relations["target_entity"].str.upper().str.replace('Ё', 'Е')

        return entities, relations