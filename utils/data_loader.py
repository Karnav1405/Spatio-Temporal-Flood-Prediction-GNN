from pathlib import Path
import importlib

import numpy as np
import pandas as pd

from graph_builder import build_kerala_district_graph


def _load_torch():
    """Try loading torch without making it a hard dependency."""
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError:
        return None


def graph_to_edge_index(graph):
    """Convert an undirected NetworkX graph to PyG-style edge_index (both directions)."""
    torch = _load_torch()
    node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}

    edges = []
    for source, target in graph.edges():
        source_idx = node_to_idx[source]
        target_idx = node_to_idx[target]
        edges.append([source_idx, target_idx])
        edges.append([target_idx, source_idx])

    if torch is not None:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = np.asarray(edges, dtype=np.int64).T

    return edge_index, list(graph.nodes())


def create_sequence_dataset(data_values: np.ndarray, seq_len: int = 5):
    """Create (X, y) where X is seq_len days and y is the next day."""
    torch = _load_torch()

    x_list = []
    y_list = []

    for start_idx in range(data_values.shape[0] - seq_len):
        end_idx = start_idx + seq_len
        x_list.append(data_values[start_idx:end_idx])
        y_list.append(data_values[end_idx])

    x_np = np.stack(x_list).astype(np.float32)
    y_np = np.stack(y_list).astype(np.float32)

    if torch is not None:
        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)
    else:
        x = x_np
        y = y_np

    return x, y


def main() -> None:
    data_path = Path("data") / "clean_flood_data.csv"
    df = pd.read_csv(data_path)

    graph = build_kerala_district_graph()
    edge_index, district_order = graph_to_edge_index(graph)

    missing_cols = [district for district in district_order if district not in df.columns]
    if missing_cols:
        raise ValueError(
            "Missing district columns in clean_flood_data.csv: " + ", ".join(missing_cols)
        )

    # Keep column order aligned with graph node indices.
    district_data = df[district_order]
    data_values = district_data.values

    x, y = create_sequence_dataset(data_values, seq_len=5)

    print(f"edge_index shape: {tuple(edge_index.shape)}")
    print(f"X shape: {tuple(x.shape)}")
    print(f"y shape: {tuple(y.shape)}")
    print("Data loader ready!")


if __name__ == "__main__":
    main()
