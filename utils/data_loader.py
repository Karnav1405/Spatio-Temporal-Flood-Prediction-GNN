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


def augment_training_data(
    x_train,
    y_train,
    augment_factor: int = 5,
    noise_level: float = 0.1,
    seed: int = 42,
):
    """Augment train data with small uniform noise while keeping targets aligned.

    Args:
        x_train: Train input sequences.
        y_train: Train targets.
        augment_factor: Total train multiplier. 5 means output has 5x samples.
        noise_level: Uniform noise range in [-noise_level, +noise_level].
        seed: Random seed for reproducibility.

    Returns:
        (x_train_aug, y_train_aug)
    """
    if augment_factor < 1:
        raise ValueError("augment_factor must be >= 1")

    torch = _load_torch()

    if torch is not None and torch.is_tensor(x_train) and torch.is_tensor(y_train):
        generator = torch.Generator().manual_seed(seed)
        x_aug_parts = [x_train]
        y_aug_parts = [y_train]

        for _ in range(augment_factor - 1):
            noise = (torch.rand(x_train.shape, generator=generator) * 2 - 1) * noise_level
            noise = noise.to(dtype=x_train.dtype, device=x_train.device)
            x_aug_parts.append(x_train + noise)
            y_aug_parts.append(y_train)

        x_train_aug = torch.cat(x_aug_parts, dim=0)
        y_train_aug = torch.cat(y_aug_parts, dim=0)
        return x_train_aug, y_train_aug

    # NumPy fallback when torch is unavailable.
    rng = np.random.default_rng(seed)
    x_aug_parts = [x_train]
    y_aug_parts = [y_train]

    for _ in range(augment_factor - 1):
        noise = rng.uniform(-noise_level, noise_level, size=x_train.shape).astype(x_train.dtype)
        x_aug_parts.append(x_train + noise)
        y_aug_parts.append(y_train)

    x_train_aug = np.concatenate(x_aug_parts, axis=0)
    y_train_aug = np.concatenate(y_aug_parts, axis=0)
    return x_train_aug, y_train_aug


def split_and_augment_train_data(
    x,
    y,
    train_ratio: float = 0.8,
    augment_factor: int = 5,
    noise_level: float = 0.1,
    seed: int = 42,
):
    """Split into train/test and augment train only.

    This keeps test data clean and unchanged while expanding train data.
    """
    torch = _load_torch()

    num_samples = x.shape[0]
    train_size = int(train_ratio * num_samples)

    if torch is not None and torch.is_tensor(x) and torch.is_tensor(y):
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(num_samples, generator=generator)
        train_idx = indices[:train_size]
        test_idx = indices[train_size:]

        x_train = x[train_idx]
        y_train = y[train_idx]
        x_test = x[test_idx]
        y_test = y[test_idx]
    else:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(num_samples)
        train_idx = indices[:train_size]
        test_idx = indices[train_size:]

        x_train = x[train_idx]
        y_train = y[train_idx]
        x_test = x[test_idx]
        y_test = y[test_idx]

    x_train_aug, y_train_aug = augment_training_data(
        x_train=x_train,
        y_train=y_train,
        augment_factor=augment_factor,
        noise_level=noise_level,
        seed=seed,
    )

    print(
        "Dataset sizes after augmentation: "
        f"train={x_train_aug.shape[0]} (original {x_train.shape[0]}), "
        f"test={x_test.shape[0]}"
    )

    return x_train_aug, y_train_aug, x_test, y_test


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
