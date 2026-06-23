from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from graph_rag import load_graph


def main() -> None:
    output = Path("outputs")
    graph = load_graph(output / "knowledge_graph.json")

    degree = dict(graph.degree())
    top_nodes = [node for node, _ in Counter(degree).most_common(35)]
    subgraph = graph.subgraph(top_nodes).copy()

    plt.figure(figsize=(16, 11))
    pos = nx.spring_layout(subgraph.to_undirected(), seed=42, k=0.8)
    node_sizes = [300 + 90 * degree.get(node, 1) for node in subgraph.nodes]

    nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes, node_color="#8ecae6", alpha=0.9)
    nx.draw_networkx_edges(subgraph, pos, edge_color="#555", alpha=0.35, arrows=False)
    nx.draw_networkx_labels(subgraph, pos, font_size=8)

    plt.title("Top Entity Graph - EV Corpus")
    plt.axis("off")
    plt.tight_layout()
    path = output / "graph_preview.png"
    plt.savefig(path, dpi=180)
    print(f"Saved: {path.resolve()}")


if __name__ == "__main__":
    main()
