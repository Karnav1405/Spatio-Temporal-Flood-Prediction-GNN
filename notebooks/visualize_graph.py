from pathlib import Path
import sys

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
UTILS_DIR = ROOT_DIR / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from graph_builder import build_kerala_district_graph


def main() -> None:
    graph = build_kerala_district_graph()

    data_path = ROOT_DIR / "data" / "clean_flood_data.csv"
    flood_df = pd.read_csv(data_path)
    flood_df["date"] = pd.to_datetime(flood_df["date"])

    peak_day = pd.Timestamp("2018-08-15")
    peak_day_rows = flood_df[flood_df["date"] == peak_day]
    if peak_day_rows.empty:
        raise ValueError("Date 2018-08-15 not found in clean_flood_data.csv")

    peak_values = peak_day_rows.iloc[0]

    severity_to_color = {
        0: "green",
        1: "yellow",
        2: "orange",
        3: "red",
    }
    severity_to_size = {
        0: 900,
        1: 1300,
        2: 1800,
        3: 2400,
    }

    node_colors = []
    node_sizes = []
    for district in graph.nodes():
        severity = int(float(peak_values[district]))
        severity = max(0, min(3, severity))
        node_colors.append(severity_to_color[severity])
        node_sizes.append(severity_to_size[severity])

    plt.figure(figsize=(14, 10))
    positions = nx.spring_layout(graph, seed=42)

    nx.draw(
        graph,
        positions,
        with_labels=True,
        node_color=node_colors,
        node_size=node_sizes,
        font_size=9,
        font_weight="bold",
        edge_color="gray",
        linewidths=1.0,
    )

    plt.title("Kerala Flood Severity Graph - Peak Day (2018-08-15)", fontsize=14)

    output_path = ROOT_DIR / "results" / "kerala_flood_peak_day.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
