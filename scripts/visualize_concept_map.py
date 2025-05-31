import json
import sys
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path


def load_concept_map(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def build_graph(concept_map):
    G = nx.DiGraph()
    for rel in concept_map["relationships"]:
        G.add_edge(rel["source"], rel["target"], label=rel["relationship"])
    return G


def visualize_graph(G, max_nodes=30, title=None):
    # If the graph is too large, show a subgraph
    if len(G.nodes) > max_nodes:
        # Get the largest connected component (undirected for visualization)
        largest_cc = max(nx.weakly_connected_components(G), key=len)
        sub_nodes = list(largest_cc)[:max_nodes]
        H = G.subgraph(sub_nodes)
    else:
        H = G
    pos = nx.spring_layout(H, k=0.5, seed=42)
    plt.figure(figsize=(12, 8))
    nx.draw(H, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=1500, font_size=10, arrows=True)
    edge_labels = nx.get_edge_attributes(H, 'label')
    nx.draw_networkx_edge_labels(H, pos, edge_labels=edge_labels, font_color='red', font_size=8)
    plt.title(title or "Concept Map Visualization")
    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/visualize_concept_map.py <concept_map.json> [max_nodes]")
        sys.exit(1)
    json_path = sys.argv[1]
    max_nodes = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    concept_map = load_concept_map(json_path)
    G = build_graph(concept_map)
    visualize_graph(G, max_nodes=max_nodes, title=Path(json_path).stem)


if __name__ == "__main__":
    main() 