# Based on https://github.com/gusye1234/nano-graphrag/blob/main/nano_graphrag/

import asyncio
from dataclasses import asdict
from typing import List

from ragu.graph.knowledge_graph import KnowledgeGraph
from ragu.graph.types import Entity

# TODO: refactor and implement other functions
async def _find_most_related_edges_from_entities(entities, knowledge_graph: KnowledgeGraph):
    all_related_edges = []
    for entity in entities:
        relations = await knowledge_graph.get_all_entity_relations(entity.id)
        all_related_edges.extend(relations)

    # all_edges_degree = [knowledge_graph.edge_degree(edge, e[1]) for edge in all_edges]
    all_edges_data = [asdict(edge) for edge in all_related_edges]
    all_edges_data = sorted(
        all_edges_data,
        key=lambda x: (x["relation_strength"]),
        reverse=True
    )

    return all_edges_data


async def _find_most_related_text_unit_from_entities(
        entities: List[Entity],
        knowledge_graph: KnowledgeGraph
):
    chunks_id = [entity.source_chunk_id for entity in entities]

    edges = []
    for entity in entities:
        edges.append(await knowledge_graph.get_all_entity_relations(entity.id))

    neighbors_candidate: List[List[Entity]] = await asyncio.gather(*[
        knowledge_graph.get_neighbors(entity.id) for entity in entities
    ])
    neighbors = sum(neighbors_candidate, [])

    all_one_hop_text_units_lookup = { neighbor.id : neighbor.source_chunk_id for neighbor in neighbors }

    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(chunks_id, edges)):
        for c_id in this_text_units:
            if c_id in all_text_units_lookup:
                continue
            relation_counts = 0
            for e in this_edges:
                if (
                        e.object_id in all_one_hop_text_units_lookup
                        and c_id in all_one_hop_text_units_lookup[e.object_id]
                ):
                    relation_counts += 1
            all_text_units_lookup[c_id] = {
                "data": await knowledge_graph.index.chunks_kv_storage.get_by_id(c_id),
                "order": index,
                "relation_counts": relation_counts,
            }
    all_text_units = [
        {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
    ]
    chunks = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )
    all_text_units = [t["data"] for t in chunks]
    return all_text_units

#
# async def _find_most_related_community_from_entities(
#         entities: List[Entity],
#         knowledge_graph: KnowledgeGraph,
#         level: int = 2
# ):
#     related_communities = set()
#     for node in entities:
#         if node.clusters:
#             related_communities.update(node.clusters)
#
#     related_communities = list(filter(lambda dp: dp["level"] <= level, related_communities))
#     related_community_data = asyncio.gather(*[
#         knowledge_graph.index.community_summary_kv_storage.get_by_id(cluster_data.get("cluster_id"))
#         for cluster_data in related_communities
#     ])
#
#     related_community_data = [
#         community for community in related_community_data if community and community.summary is not None
#     ]
#
#     related_community_data = sorted(
#         related_community_data,
#         key=lambda x: x["community_report"].get("rating", -1),
#         reverse=True
#     )
#
#     sorted_community_datas = [
#         combine_report_text(community["community_report"]) for community in related_community_data
#     ]
#
#     return sorted_community_datas
