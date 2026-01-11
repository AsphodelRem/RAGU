"""
Knowledge Graph Visualization Script using PyVis.

This script loads a RAGU knowledge graph from storage files and creates
an interactive HTML visualization showing entities, relations, and metadata.

Usage:
    python visualize_knowledge_graph.py [--graph-path PATH] [--output OUTPUT]

Examples:
    # Use default path (./ragu_working_dir/*/knowledge_graph.gml)
    python visualize_knowledge_graph.py

    # Specify a custom graph file
    python visualize_knowledge_graph.py --graph-path ./my_data/knowledge_graph.gml

    # Specify output file
    python visualize_knowledge_graph.py --output my_graph.html
"""

import argparse
import colorsys
import json
import os
from glob import glob
from typing import Dict, List, Optional

import networkx as nx
from pyvis.network import Network


DEFAULT_NODE_COLOR = "#BDC3C7"

def generate_distinct_colors(n: int) -> List[str]:
    """
    Generate n visually distinct colors using golden ratio distribution in HSL space.
    """
    colors = []
    golden_ratio = 0.618033988749895

    for i in range(n):
        hue = (i * golden_ratio) % 1.0
        # Vary saturation and lightness slightly for more distinction
        saturation = 0.65 + (i % 3) * 0.1
        lightness = 0.5 + (i % 2) * 0.1

        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )
        colors.append(hex_color)

    return colors


def build_entity_color_map(entity_types: set) -> Dict[str, str]:
    """
    Build a color map for a set of entity types.
    """
    sorted_types = sorted(entity_types)
    colors = generate_distinct_colors(len(sorted_types))
    return {entity_type: colors[i] for i, entity_type in enumerate(sorted_types)}


def get_latest_graph_path() -> Optional[str]:
    """
    Find the most recent knowledge graph file in the default working directory.
    """
    base_path = os.path.join(os.getcwd(), "ragu_working_dir")
    if not os.path.exists(base_path):
        return None

    pattern = os.path.join(base_path, "*", "knowledge_graph.gml")
    graph_files = glob(pattern)

    if not graph_files:
        return None

    # Sort by modification time, get the most recent
    graph_files.sort(key=os.path.getmtime, reverse=True)
    return graph_files[0]


def load_graph(graph_path: str) -> nx.Graph:
    """
    Load a NetworkX graph from a GML file.
    """
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph file not found: {graph_path}")

    return nx.read_gml(graph_path)


def load_chunks(graph_dir: str) -> Dict[str, dict]:
    """
    Load chunk data from KV storage if available.
    """
    chunks_path = os.path.join(graph_dir, "kv_chunks.json")
    if os.path.exists(chunks_path):
        with open(chunks_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_node_color(entity_type: str, color_map: Dict[str, str]) -> str:
    """
    Get color for an entity type from the provided color map.
    """
    return color_map.get(entity_type, DEFAULT_NODE_COLOR)


def truncate_text(text: str, max_length: int = 200) -> str:
    """
    Truncate text to a maximum length with ellipsis.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def format_node_title(node_id: str, attrs: dict, chunks: Dict[str, dict]) -> str:
    """
    Format the hover tooltip for a node.
    """
    lines = [
        f"<b>ID:</b> {node_id}",
        f"<b>Name:</b> {attrs.get('entity_name', 'Unknown')}",
        f"<b>Type:</b> {attrs.get('entity_type', 'Unknown')}",
    ]

    description = attrs.get("description", "")
    if description:
        lines.append(f"<b>Description:</b> {truncate_text(description)}")

    source_chunks = attrs.get("source_chunk_id", [])
    if source_chunks:
        if isinstance(source_chunks, str):
            source_chunks = [source_chunks]
        lines.append(f"<b>Source Chunks:</b> {len(source_chunks)}")

    clusters = attrs.get("clusters", [])
    if clusters:
        if isinstance(clusters, str):
            try:
                clusters = eval(clusters)  # Parse string representation of list
            except:
                clusters = []
        if clusters:
            cluster_info = ", ".join(
                f"L{c.get('level', '?')}-C{c.get('cluster_id', '?')}"
                for c in clusters[:5]
            )
            lines.append(f"<b>Clusters:</b> {cluster_info}")

    return "<br>".join(lines)


def format_edge_title(edge_data: dict) -> str:
    """
    Format the hover tooltip for an edge.
    """
    lines = []

    description = edge_data.get("description", "")
    if description:
        lines.append(f"<b>Relation:</b> {truncate_text(description)}")

    strength = edge_data.get("relation_strength", 1.0)
    lines.append(f"<b>Strength:</b> {strength:.2f}")

    source_chunks = edge_data.get("source_chunk_id", [])
    if source_chunks:
        if isinstance(source_chunks, str):
            source_chunks = [source_chunks]
        lines.append(f"<b>Source Chunks:</b> {len(source_chunks)}")

    rel_id = edge_data.get("id", "")
    if rel_id:
        lines.append(f"<b>ID:</b> {rel_id}")

    return "<br>".join(lines) if lines else "Relation"


def create_visualization(
    graph: nx.Graph,
    chunks: Dict[str, dict],
    output_path: str,
    height: str = "900px",
    width: str = "100%",
    bgcolor: str = "#222222",
    font_color: str = "white",
) -> None:
    """Create an interactive PyVis visualization of the knowledge graph."""

    # Build color map for all entity types in the graph
    entity_types = set()
    for node_id in graph.nodes():
        entity_type = graph.nodes[node_id].get("entity_type", "Unknown")
        entity_types.add(entity_type)
    color_map = build_entity_color_map(entity_types)

    net = Network(
        height=height,
        width=width,
        bgcolor=bgcolor,
        font_color=font_color,
        directed=False,
        notebook=False,
        select_menu=True,
        filter_menu=True,
    )

    # Configure physics for better layout
    net.set_options("""
    {
        "nodes": {
            "font": {
                "size": 14,
                "face": "arial"
            },
            "borderWidth": 2,
            "borderWidthSelected": 4
        },
        "edges": {
            "color": {
                "inherit": false
            },
            "smooth": {
                "type": "continuous",
                "forceDirection": "none"
            },
            "font": {
                "size": 10,
                "align": "middle"
            }
        },
        "physics": {
            "enabled": true,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 150,
                "springConstant": 0.08,
                "damping": 0.4
            },
            "stabilization": {
                "enabled": true,
                "iterations": 200,
                "updateInterval": 25
            }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "hideEdgesOnDrag": true,
            "hideEdgesOnZoom": true
        }
    }
    """)

    degrees = dict(graph.degree()) # type: ignore
    max_degree = max(degrees.values()) if degrees else 1
    min_size = 15
    max_size = 50

    for node_id in graph.nodes():
        attrs = dict(graph.nodes[node_id])
        entity_name = attrs.get("entity_name", str(node_id))
        entity_type = attrs.get("entity_type", "Unknown")

        degree = degrees.get(node_id, 1)
        size = min_size + (max_size - min_size) * (degree / max_degree)

        color = get_node_color(entity_type, color_map)

        title = format_node_title(node_id, attrs, chunks)

        net.add_node(
            node_id,
            label=entity_name,
            title=title,
            size=size,
            color=color,
            shape="dot",
            group=entity_type,
        )

    for source, target, edge_data in graph.edges(data=True):
        title = format_edge_title(edge_data)

        strength = float(edge_data.get("relation_strength", 1.0))
        width = 1 + min(strength * 2, 5)

        description = edge_data.get("description", "")
        label = truncate_text(description, 30) if description else ""

        net.add_edge(
            source,
            target,
            title=title,
            width=width,
            label=label,
            color="#888888",
        )

    net.save_graph(output_path)

    add_legend_to_html(output_path, graph, color_map)

    print(f"Visualization saved to: {output_path}")
    print(f"  Nodes: {graph.number_of_nodes()}")
    print(f"  Edges: {graph.number_of_edges()}")


def add_legend_to_html(html_path: str, graph: nx.Graph, color_map: Dict[str, str]) -> None:
    """
    Add a legend showing entity type colors to the HTML file.
    """

    legend_items = []
    for entity_type in sorted(color_map.keys()):
        color = color_map.get(entity_type, DEFAULT_NODE_COLOR)
        legend_items.append(
            f'<div style="display: flex; align-items: center; margin: 3px 0;">'
            f'<span style="display: inline-block; width: 16px; height: 16px; '
            f'background-color: {color}; border-radius: 50%; margin-right: 8px;"></span>'
            f'<span>{entity_type}</span></div>'
        )

    legend_html = f'''
    <div id="legend" style="
        position: absolute;
        top: 10px;
        right: 10px;
        background: rgba(0, 0, 0, 0.8);
        padding: 15px;
        border-radius: 8px;
        color: white;
        font-family: Arial, sans-serif;
        font-size: 12px;
        max-height: 400px;
        overflow-y: auto;
        z-index: 1000;
    ">
        <div style="font-weight: bold; margin-bottom: 10px; font-size: 14px;">Entity Types</div>
        {''.join(legend_items)}
        <hr style="border-color: #444; margin: 10px 0;">
        <div style="font-size: 11px; color: #aaa;">
            <div>Nodes: {graph.number_of_nodes()}</div>
            <div>Edges: {graph.number_of_edges()}</div>
            <div style="margin-top: 5px;">Node size = degree</div>
            <div>Edge width = strength</div>
        </div>
    </div>
    '''

    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    html_content = html_content.replace("</body>", legend_html + "</body>")

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a RAGU knowledge graph using PyVis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--graph-path",
        type=str,
        default=None,
        help="Path to the knowledge_graph.gml file. If not specified, uses the most recent graph in ./ragu_working_dir/"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="knowledge_graph_visualization.html",
        help="Output HTML file path (default: knowledge_graph_visualization.html)"
    )
    parser.add_argument(
        "--height",
        type=str,
        default="900px",
        help="Height of the visualization (default: 900px)"
    )
    parser.add_argument(
        "--width",
        type=str,
        default="100%",
        help="Width of the visualization (default: 100%%)"
    )
    parser.add_argument(
        "--light-theme",
        action="store_true",
        help="Use light theme instead of dark theme"
    )

    args = parser.parse_args()

    graph_path = args.graph_path
    if graph_path is None:
        graph_path = get_latest_graph_path()
        if graph_path is None:
            print("Error: No knowledge graph found in ./ragu_working_dir/")
            print("Please specify --graph-path or run a graph extraction first.")
            return 1
        print(f"Using graph: {graph_path}")

    try:
        graph = load_graph(graph_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error loading graph: {e}")
        return 1

    if graph.number_of_nodes() == 0:
        print("Warning: The graph is empty (no nodes)")
        return 1

    graph_dir = os.path.dirname(graph_path)
    chunks = load_chunks(graph_dir)

    if args.light_theme:
        bgcolor = "#ffffff"
        font_color = "#333333"
    else:
        bgcolor = "#222222"
        font_color = "white"

    create_visualization(
        graph=graph,
        chunks=chunks,
        output_path=args.output,
        height=args.height,
        width=args.width,
        bgcolor=bgcolor,
        font_color=font_color,
    )

    return 0


if __name__ == "__main__":
    exit(main())
