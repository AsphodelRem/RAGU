"""
Knowledge Graph to Neo4j Export Script.

This script loads a RAGU knowledge graph from GML format and imports it into Neo4j
for visualization and querying. Optionally exports to GraphML format.

Usage:
    python gml_to_neo4j.py [--graph-path PATH] [--neo4j-uri URI] [--clear]

Examples:
    # Use default paths and Neo4j connection
    python gml_to_neo4j.py

    # Specify custom graph and Neo4j credentials
    python gml_to_neo4j.py --graph-path ./my_data/knowledge_graph.gml \\
        --neo4j-uri bolt://localhost:7687 --neo4j-user neo4j --neo4j-password secret

    # Clear existing data before import
    python gml_to_neo4j.py --clear

Requirements:
    pip install networkx neo4j

This script was made by Claude Code
"""

import argparse
import os
from typing import Dict, Any, List

import networkx as nx

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None


def load_gml_graph(graph_path: str) -> nx.Graph:
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    return nx.read_gml(graph_path)


def sanitize_property(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        # Neo4j supports lists of primitives
        if all(isinstance(v, (str, int, float, bool)) for v in value):
            return value
        return str(value)
    return str(value)


def sanitize_properties(props: Dict[str, Any]) -> Dict[str, Any]:
    return {k: sanitize_property(v) for k, v in props.items() if v is not None}


class Neo4jImporter:
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
    ):
        if GraphDatabase is None:
            raise ImportError("neo4j package is required. Install with: pip install neo4j")

        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._verify_connection()

    def _verify_connection(self) -> None:
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
            print("Connected to Neo4j successfully")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Neo4j: {e}")

    def close(self) -> None:
        self.driver.close()

    def clear_database(self) -> None:
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("Cleared all existing data from Neo4j")

    def create_index(self) -> None:
        with self.driver.session() as session:
            try:
                # Neo4j 5.x syntax
                session.run("CREATE INDEX entity_id_idx IF NOT EXISTS FOR (e:Entity) ON (e.id)")
            except Exception:
                try:
                    # Neo4j 4.x syntax
                    session.run("CREATE INDEX ON :Entity(id)")
                except Exception:
                    pass

    def import_graph(self, graph: nx.Graph, batch_size: int = 500) -> None:
        """
        Import NetworkX graph into Neo4j.

        :param graph: NetworkX graph to import.
        :param batch_size: Number of nodes/edges to import per transaction.
        """
        self.create_index()

        # Import nodes
        nodes_data = []
        for node_id in graph.nodes():
            attrs = dict(graph.nodes[node_id])
            node_data = {
                "id": str(node_id),
                "entity_name": attrs.get("entity_name", str(node_id)),
                "entity_type": attrs.get("entity_type", "UNKNOWN"),
                **sanitize_properties(attrs)
            }
            nodes_data.append(node_data)

        print(f"Importing {len(nodes_data)} nodes...")
        self._batch_create_nodes(nodes_data, batch_size)

        # Import edges
        edges_data = []
        for source, target, edge_attrs in graph.edges(data=True):
            # Separate source/target for matching from properties
            props = sanitize_properties(edge_attrs)
            props["description"] = edge_attrs.get("description", "")
            props["relation_strength"] = float(edge_attrs.get("relation_strength", 1.0))
            # Remove source/target from props if present (they're for matching only)
            props.pop("source", None)
            props.pop("target", None)

            edge_data = {
                "source": str(source),
                "target": str(target),
                "props": props,
            }
            edges_data.append(edge_data)

        print(f"Importing {len(edges_data)} relationships...")
        self._batch_create_edges(edges_data, batch_size)

        with self.driver.session() as session:
            result = session.run("MATCH (n:Entity) RETURN count(n) as nodes")
            actual_nodes = result.single()["nodes"]
            result = session.run("MATCH ()-[r:RELATES_TO]->() RETURN count(r) as edges")
            actual_edges = result.single()["edges"]

        print(f"Import complete:")
        print(f"  Expected: {len(nodes_data)} nodes, {len(edges_data)} relationships")
        print(f"  Actual:   {actual_nodes} nodes, {actual_edges} relationships")

    def _batch_create_nodes(self, nodes: List[Dict], batch_size: int) -> None:
        with self.driver.session() as session:
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i:i + batch_size]
                session.run(
                    """
                    UNWIND $nodes AS node
                    CREATE (e:Entity)
                    SET e = node
                    """,
                    nodes=batch
                )

    def _batch_create_edges(self, edges: List[Dict], batch_size: int) -> None:
        """Create relationships in batches."""
        with self.driver.session() as session:
            for i in range(0, len(edges), batch_size):
                batch = edges[i:i + batch_size]
                result = session.run(
                    """
                    UNWIND $edges AS edge
                    MATCH (s:Entity {id: edge.source})
                    MATCH (t:Entity {id: edge.target})
                    CREATE (s)-[r:RELATES_TO]->(t)
                    SET r = edge.props
                    RETURN count(r) as created
                    """,
                    edges=batch
                )
                record = result.single()
                if record:
                    created = record["created"]
                    if created < len(batch):
                        print(f"  Warning: Only created {created}/{len(batch)} edges in batch")


def main():
    parser = argparse.ArgumentParser(
        description="Export RAGU knowledge graph to Neo4j and/or GraphML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--graph-path",
        type=str,
        default=None,
        help="Path to .gml"
    )
    parser.add_argument(
        "--neo4j-uri",
        type=str,
        default="bolt://localhost:7687",
        help="Neo4j connection URI (default: bolt://localhost:7687)"
    )
    parser.add_argument(
        "--neo4j-user",
        type=str,
        default="neo4j",
        help="Neo4j username (default: neo4j)"
    )
    parser.add_argument(
        "--neo4j-password",
        type=str,
        default="password",
        help="Neo4j password (default: password)"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing Neo4j data before import"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for Neo4j imports (default: 500)"
    )

    args = parser.parse_args()

    try:
        graph = load_gml_graph(args.graph_path)
        print(f"Loaded graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error loading graph: {e}")
        return 1

    if graph.number_of_nodes() == 0:
        print("Warning: The graph is empty (no nodes)")
        return 1

    try:
        importer = Neo4jImporter(
            uri=args.neo4j_uri,
            user=args.neo4j_user,
            password=args.neo4j_password,
        )
    except ImportError as e:
        print(f"Error: {e}")
        return 1
    except ConnectionError as e:
        print(f"Error: {e}")
        return 1

    try:
        if args.clear:
            importer.clear_database()

        importer.import_graph(graph, batch_size=args.batch_size)

        print("\nNeo4j Browser queries to try:")
        print("  MATCH (n) RETURN n LIMIT 100")
        print("  MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50")
        print("  MATCH (n:Entity) WHERE n.entity_type = 'PERSON' RETURN n")
    finally:
        importer.close()

    return 0


if __name__ == "__main__":
    exit(main())
