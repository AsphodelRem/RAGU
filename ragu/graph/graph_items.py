import pandas as pd
from tqdm import tqdm
from typing import Any, List, Dict, Union

from ragu.common.settings import settings
from ragu.common.types import Node, Relation
from ragu.utils.triplet_parser import parse_description


def get_nodes(entities: pd.DataFrame) -> List[Node]:
    nodes: list[Node] = []
        
    for i, row in enumerate(entities.iterrows()):
        nodes.append(
            Node(
                id=i, 
                entity=row['entity'], 
                description=row['description']
            )
        )

    return nodes
    

def get_relationships(relations: pd.DataFrame, nodes: List[Node]) -> List[Relation]:
    relationships: List[Relation] = []
    print(nodes)

    for i, row in enumerate(relations.iterrows()):
        row = row[1]
        source_node = nodes[nodes.index(row['source'])]
        target_node = nodes[nodes.index(row['target'])]
        relation = row['relation']

        relationships.append(
            Relation(
                source=source_node, 
                target=target_node, 
                description=relation
            )
        )

    return relationships


class EntityExtractor:
    @staticmethod
    def extract(raw_data: List[Dict[str, str]], client: Any) -> pd.DataFrame:
        """
        Extracts, merges, and summarizes entity descriptions from raw data.
        """
        entities = EntityExtractor.get_description(raw_data, client)
        return EntityExtractor.summarize(entities, client)
    
    @staticmethod
    def summarize(triplets: pd.DataFrame, client: Any) -> pd.DataFrame:
        """
        Summarizes descriptions of entities using an LLM client.
        """
        from ragu.utils.default_prompts.entites_prompts import (
            entities_description_summary_prompt
        )

        summaries = []
        
        for _, row in tqdm(triplets.iterrows(), desc='Summarizing entity descriptions'):
            entity, description = row['entity'], row['description']
            if not entity or not description:
                continue
            
            text = f"Сущность: {entity}\nОписание: {description}"
            response = client.chat.completions.create(
                model=settings.llm_model_name,
                messages=[
                    {"role": "system", "content": entities_description_summary_prompt},
                    {"role": "user", "content": text}
                ]
            )
            summaries.append({
                'entity': entity, 
                'summarized_description': response.choices[0].message.content
            })
        
        summarization_table = pd.DataFrame(summaries)
        summarization_table.to_csv('summarization_table.csv', index=False)
        return summarization_table
    
    @staticmethod
    def merge_entities(entities: pd.DataFrame) -> pd.DataFrame:
        """
        Groups entities by name, aggregating descriptions into a single string.
        """
        return entities.groupby("entity")["description"].apply("".join).reset_index()
    
    @staticmethod
    def get_description(raw_data: List[Dict[str, str]], client: Any) -> pd.DataFrame:
        """
        Extracts and combines descriptions for entities from raw data using an LLM client.
        """
        from ragu.utils.default_prompts.entites_prompts import (
            entites_info_prompt
        )

        report_data = []
        
        for elem in tqdm(raw_data, desc='Extracting entity descriptions'):
            relation, text_chunk = elem.get('relations', ''), elem.get('chunk', '')
            print(relation, text_chunk)
            if not relation or not text_chunk:
                continue
            
            text = f"Триплеты: {relation}\nТекст: {text_chunk}"
            response = client.chat.completions.create(
                model=settings.llm_model_name,
                messages=[
                    {"role": "system", "content": entites_info_prompt},
                    {"role": "user", "content": text}
                ]
            )
            parsed_response = parse_description(response.choices[0].message.content)
            if parsed_response:
                report_data.extend(parsed_response)
        
        report = pd.DataFrame(report_data, columns=['entity', 'description'])
        report.to_csv('report.csv', index=False)
        return EntityExtractor.merge_entities(report)


class RelationExtractor:
    @staticmethod
    def extract(triplets: List[Dict[str, str]], client: Any) -> pd.DataFrame:
        """
        Extracts and merges relationships from triplets.
        """
        return RelationExtractor.merge_relationships(triplets)
    
    @staticmethod
    def merge_relationships(
        relationships: Union[pd.DataFrame, List[Dict[str, str]]]
        ) -> pd.DataFrame:
        """
        Groups relationships by source and target entities, aggregating relations into a list.
        """
        df = pd.DataFrame(relationships, columns=["source", "relation", "target"]) \
            if not isinstance(relationships, pd.DataFrame) else relationships
        return df.groupby(["source", "target"]).agg({"relation": list}).reset_index()
