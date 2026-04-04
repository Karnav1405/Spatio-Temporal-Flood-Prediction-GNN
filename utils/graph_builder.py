from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


def build_kerala_district_graph() -> nx.Graph:
    """Create an undirected graph of Kerala districts and shared borders."""
    districts = [
        "Thiruvananthapuram",
        "Kollam",
        "Pathanamthitta",
        "Alappuzha",
        "Kottayam",
        "Ernakulam",
        "Idukki",
        "Thrissur",
        "Palakkad",
        "Malappuram",
        "Wayanad",
        "Kozhikkode",
        "Kannur",
        "Kasaragod",
    ]

    border_map = {
        "Thiruvananthapuram": ["Kollam"],
        "Kollam": ["Thiruvananthapuram", "Pathanamthitta", "Alappuzha"],
        "Pathanamthitta": ["Kollam", "Alappuzha", "Kottayam", "Idukki"],
        "Alappuzha": ["Kollam", "Pathanamthitta", "Kottayam", "Ernakulam"],
        "Kottayam": ["Pathanamthitta", "Alappuzha", "Ernakulam", "Idukki"],
        "Ernakulam": ["Alappuzha", "Kottayam", "Idukki", "Thrissur"],
        "Idukki": ["Pathanamthitta", "Kottayam", "Ernakulam", "Thrissur", "Wayanad"],
        "Thrissur": ["Ernakulam", "Idukki", "Palakkad", "Malappuram"],
        "Palakkad": ["Thrissur", "Malappuram", "Wayanad"],
        "Malappuram": ["Thrissur", "Palakkad", "Wayanad", "Kozhikkode"],
        "Wayanad": ["Idukki", "Palakkad", "Malappuram", "Kozhikkode", "Kannur"],
        "Kozhikkode": ["Malappuram", "Wayanad", "Kannur"],
        "Kannur": ["Kozhikkode", "Wayanad", "Kasaragod"],
        "Kasaragod": ["Kannur"],
    }

    graph = nx.Graph()
    graph.add_nodes_from(districts)

    for district, neighbors in border_map.items():
        for neighbor in neighbors:
            graph.add_edge(district, neighbor)

    return graph


def visualize_graph(graph: nx.Graph, output_path: Path) -> None:
    """Visualize and save the district graph with high-risk districts highlighted."""
    high_risk_districts = {"Idukki", "Wayanad", "Malappuram"}
    node_colors = [
        "red" if node in high_risk_districts else "dodgerblue" for node in graph.nodes()
    ]

    plt.figure(figsize=(14, 10))
    positions = nx.spring_layout(graph, seed=42)

    nx.draw(
        graph,
        positions,
        with_labels=True,
        node_color=node_colors,
        node_size=1700,
        font_size=9,
        font_weight="bold",
        edge_color="gray",
        linewidths=1.0,
    )

    plt.title("Kerala District Border Graph for Flood Prediction", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    graph = build_kerala_district_graph()
    output_file = Path("results") / "kerala_district_graph.png"
    visualize_graph(graph, output_file)

    print(f"Number of nodes: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")


if __name__ == "__main__":
    main()
