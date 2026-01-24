# Based on https://github.com/gusye1234/nano-graphrag/blob/main/nano_graphrag/

import asyncio
from dataclasses import asdict
from typing import List

from ragu.common.prompts.default_models import SubQuery
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

async def _find_documents_id(entities: List[Entity]):
    documents_set = set()
    for entity in entities:
        if hasattr(entity, 'documents_id') and entity.documents_id:
            documents_set.update(entity.documents_id)
    return list(documents_set)


async def _find_most_related_community_from_entities(
        entities: List[Entity],
        knowledge_graph: KnowledgeGraph,
        level: int = 2
):
    if not entities:
        return []

    desired_pairs: set[tuple[int, int]] = set()
    for entity in entities:
        if not getattr(entity, "clusters", None):
            continue
        for cluster_data in entity.clusters:
            try:
                c_level = int(cluster_data.get("level", 9999))
            except Exception:
                continue
            if c_level <= level:
                cid = cluster_data.get("cluster_id")
                if cid is not None:
                    desired_pairs.add((c_level, int(cid)))

    if not desired_pairs:
        return []

    community_store = knowledge_graph.index.community_kv_storage
    summary_store = knowledge_graph.index.community_summary_kv_storage

    all_comm_keys = await community_store.all_keys()
    if not all_comm_keys:
        return []

    all_comm_data = await community_store.get_by_ids(all_comm_keys)

    matched_ids: list[str] = []
    for key, data in zip(all_comm_keys, all_comm_data):
        if not data:
            continue
        try:
            comm_level = int(data.get("level", -1))
            comm_cid = int(data.get("cluster_id", -1))
        except Exception:
            continue
        if (comm_level, comm_cid) in desired_pairs:
            matched_ids.append(key)

    if not matched_ids:
        return []

    summaries = await summary_store.get_by_ids(matched_ids)
    final_summaries = [s for s in summaries if s]

    return final_summaries

def _topological_sort(subqueries: List[SubQuery]) -> List[SubQuery]:
    by_id = {q.id: q for q in subqueries}
    visited = set()
    ordered: List[SubQuery] = []

    def visit(q: SubQuery):
        if q.id in visited:
            return
        for dep in q.depends_on:
            visit(by_id[dep])
        visited.add(q.id)
        ordered.append(q)

    for q in subqueries:
        visit(q)

    return ordered