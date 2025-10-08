from __future__ import annotations

import os
import asyncio
from collections import defaultdict
from dataclasses import asdict
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Hashable
)

import networkx as nx
from graspologic.partition import HierarchicalClusters, hierarchical_leiden
from graspologic.utils import largest_connected_component

from ragu.graph.types import Entity, Relation, Community
from ragu.storage.base_storage import BaseGraphStorage


def _entity_to_attrs(e: Entity) -> Dict[str, Any]:
    return dict(
        entity_name=e.entity_name,
        entity_type=e.entity_type,
        description=e.description,
        source_chunk_id=list(e.source_chunk_id),
        documents_id=list(e.documents_id),
        clusters=list(e.clusters),
    )


def _attrs_to_entity(node_id: str, d: Dict[str, Any]) -> Entity:
    return Entity(
        id=node_id,
        entity_name=d.get("entity_name", str(node_id)),
        entity_type=d.get("entity_type", "Unknown"),
        description=d.get("description", ""),
        source_chunk_id=list(d.get("source_chunk_id", [])),
        documents_id=list(d.get("documents_id", [])),
        clusters=list(d.get("clusters", [])),
    )


class NetworkXStorage(BaseGraphStorage):
    """
    NetworkX-based implementation of :class:`BaseGraphStorage`.

    This class provides a lightweight, file-backed storage interface for
    entities and relations using NetworkX as the underlying graph structure.
    Supports node/edge CRUD operations, clustering, and persistence.
    """

    def __init__(
        self,
        filename: str,
        clustering_params=None,
        **kwargs,
    ):
        """
        Initialize a new :class:`NetworkXStorage`.

        :param filename: Path to a `.gml` file used for persistence.
        :param clustering_params: Optional parameters for community detection.
        """
        if clustering_params is None:
            clustering_params = {"max_community_size": 1000}

        self._graph: nx.Graph = nx.read_gml(filename) if os.path.exists(filename) else nx.Graph()
        self._where_to_save = filename
        self._clustering_params = clustering_params

    async def index_done_callback(self) -> None:
        """
        Persist the current graph state to disk in GML format.
        """
        nx.write_gml(self._graph, self._where_to_save)

    async def query_done_callback(self) -> None:
        """
        Callback executed after a query is completed.
        Reserved for potential post-processing hooks.
        """
        pass

    async def index_start_callback(self) -> None:
        """
        Callback executed before indexing starts.
        Reserved for potential setup hooks.
        """
        pass

    async def get_node_edges(self, source_node_id: str) -> List[Relation]:
        """
        Retrieve all edges connected to a given node.

        Each returned :class:`Relation` includes associated metadata
        and node display names when available. Missing nodes are tolerated.

        :param source_node_id: ID of the node whose edges to fetch.
        :return: List of relations connected to the node.
        """
        if not self._graph.has_node(source_node_id):
            return []

        relations: List[Relation] = []
        for u, v, metadata in self._graph.edges(source_node_id, data=True):
            subject_id = str(u)
            object_id = str(v)
            subject_name = self._graph.nodes.get(u, {}).get("entity_name", subject_id)
            object_name = self._graph.nodes.get(v, {}).get("entity_name", object_id)
            relation = Relation(
                subject_id=subject_id,
                object_id=object_id,
                subject_name=subject_name,
                object_name=object_name,
                description=metadata.get("description", ""),
                relation_strength=float(metadata.get("relation_strength", 1.0)),
                source_chunk_id=list(metadata.get("source_chunk_id", [])),
                id=metadata.get("id"),
            )
            relations.append(relation)

        seen: set[Tuple[str, str]] = set()
        unique_relations: List[Relation] = []
        for r in relations:
            key = tuple(sorted((r.subject_id, r.object_id)))
            if key in seen:
                continue
            seen.add(key)
            unique_relations.append(r)
        return unique_relations

    # TODO: add calculating
    async def get_edge_degree(self, source_node_id: str, target_node_id: str) -> int:
        """
        Return the degree or strength of a specific edge.

        :param source_node_id: Source node ID.
        :param target_node_id: Target node ID.
        :return: Edge degree or weight (0 if not defined).
        """
        return 0

    async def upsert_edge(self, edge: Relation) -> None:
        """
        Insert or update an edge in the graph.

        :param edge: The relation to add or update.
        """
        edge_data = asdict(edge)
        edge_data.pop("subject_id", None)
        edge_data.pop("object_id", None)
        self._graph.add_edge(edge.subject_id, edge.object_id, **edge_data)

    async def get_edge(self, source_node_id: str, target_node_id: str) -> Relation | None:
        """
        Retrieve a single edge between two nodes if it exists.

        :param source_node_id: Source node ID.
        :param target_node_id: Target node ID.
        :return: Relation object or ``None`` if no edge exists.
        """
        if not self._graph.has_edge(source_node_id, target_node_id):
            return None

        data = self._graph.edges[source_node_id, target_node_id]
        return Relation(
            source_node_id,
            target_node_id,
            **data
        )

    async def has_node(self, node_id: Hashable) -> bool:
        """
        Check whether a node exists in the graph.

        :param node_id: Node identifier.
        :return: ``True`` if node exists, otherwise ``False``.
        """
        return self._graph.has_node(node_id)

    async def has_edge(self, src: Hashable, dst: Hashable) -> bool:
        """
        Check whether an edge exists between two nodes.

        :param src: Source node ID.
        :param dst: Destination node ID.
        :return: ``True`` if edge exists, otherwise ``False``.
        """
        return self._graph.has_edge(src, dst)

    async def node_degree(self, node_id: Hashable) -> int:
        """
        Return the degree (number of adjacent edges) of a node.

        :param node_id: Node identifier.
        :return: Node degree as integer.
        """
        return int(self._graph.degree(node_id))  # type: ignore

    async def get_node(self, node_id: str) -> Optional[Entity]:
        """
        Retrieve a node as an :class:`Entity` object.

        :param node_id: Node identifier.
        :return: The corresponding entity or ``None`` if not found.
        """
        if not self._graph.has_node(node_id):
            return None
        data = self._graph.nodes[node_id]
        return Entity(id=node_id, **data)

    async def neighbors(self, node_id: Hashable, rel_types: Optional[List[str]] = None) -> List[Entity]:
        """
        Return all neighboring nodes (entities) connected to a given node.

        :param node_id: Node identifier.
        :type node_id: Hashable
        :param rel_types: Optional list of relation types to filter edges.
        :type rel_types: list[str] | None
        :return: List of neighboring entities.
        :rtype: list[Entity]
        """
        out: List[Entity] = []
        if not self._graph.has_node(node_id):
            return out
        for nbr in self._graph.neighbors(node_id):
            if rel_types is not None:
                d = self._graph.edges[node_id, nbr]
                if d.get("type") not in rel_types:
                    continue
            out.append(_attrs_to_entity(nbr, self._graph.nodes[nbr]))
        return out

    async def upsert_node(self, node: Entity) -> None:
        """
        Insert or update a node (entity) in the graph.

        :param node: Entity object to insert or update.
        :type node: Entity
        """
        attrs = _entity_to_attrs(node)
        self._graph.add_node(node.id, **attrs)

    async def upsert_nodes(self, nodes: Iterable[Entity]) -> None:
        """
        Insert or update multiple nodes in the graph.

        :param nodes: Iterable of entities to process.
        :type nodes: Iterable[Entity]
        """
        for n in nodes:
            await self.upsert_node(n)

    async def remove_node(self, node_id: Hashable, cascade: bool = True) -> None:
        """
        Remove a node from the graph.

        :param node_id: Node identifier.
        :type node_id: Hashable
        :param cascade: Whether to also remove connected edges (default: True).
        :type cascade: bool
        """
        if self._graph.has_node(node_id):
            self._graph.remove_node(node_id)

    async def remove_edge(self, src: Hashable, dst: Hashable) -> None:
        """
        Remove an edge between two nodes.

        :param src: Source node ID.
        :type src: Hashable
        :param dst: Destination node ID.
        :type dst: Hashable
        """
        if self._graph.has_edge(src, dst):
            self._graph.remove_edge(src, dst)

    async def remove_isolated_nodes(self) -> None:
        """
        Remove nodes with no edges from the graph.
        """
        for node in list(self._graph.nodes):
            if self._graph.degree(node) == 0:
                self._graph.remove_node(node)

    async def connected_components(self, mode: str = "weak") -> List[List[Hashable]]:
        """
        Compute connected components of the graph.

        :param mode: Component mode, e.g., ``"weak"`` or ``"strong"`` (ignored for undirected graphs).
        :type mode: str
        :return: List of connected components, each as a list of node IDs.
        :rtype: list[list[Hashable]]
        """
        comps = nx.connected_components(self._graph)
        return [list(c) for c in comps]

    async def take_only_largest_component(self) -> "NetworkXStorage":
        """
        Retain only the largest connected component in the graph.

        :return: The current storage instance with filtered graph.
        """
        self._graph = largest_connected_component(self._graph)
        return self

    async def cluster(self) -> list[Community]:
        """
        Perform hierarchical Leiden clustering to identify communities in the graph.

        :return: List of detected communities.
        """
        community_mapping: HierarchicalClusters = hierarchical_leiden(
            self._graph,
            **self._clustering_params
        )

        clusters = defaultdict(lambda: defaultdict(lambda: {"nodes": set(), "edges": set()}))
        node_membership = defaultdict(set)  # node -> {(level, cluster_id)}

        for part in community_mapping:
            level = part.level
            cid = part.cluster
            node = part.node

            self._graph.nodes[node]["clusters"].append({"level": level, "cluster_id": cid})
            clusters[level][cid]["nodes"].add(node)
            node_membership[node].add((level, cid))

        for u, v in self._graph.edges:
            common = node_membership[u].intersection(node_membership[v])
            if not common:
                continue

            a, b = (u, v) if u <= v else (v, u)
            for level, cid in common:
                clusters[level][cid]["edges"].add((a, b))

        communities: List[Community] = []
        for level, buckets in clusters.items():
            for cid, payload in buckets.items():
                nodes = payload["nodes"]
                edges = payload["edges"]

                entities = await asyncio.gather(*[self.get_node(n) for n in nodes])
                entities = [node for node in entities if node is not None]

                relations = await asyncio.gather(*[self.get_edge(source, target) for (source, target) in edges])
                relations = [relation for relation in relations if relation is not None]

                communities.append(
                    Community(
                        entities=entities,
                        relations=relations,
                        level=level,
                        cluster_id=cid,
                    )
                )

        return communities
