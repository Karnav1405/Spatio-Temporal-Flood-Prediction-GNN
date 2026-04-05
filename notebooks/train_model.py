from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure local project imports work when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
UTILS_DIR = PROJECT_ROOT / "utils"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from models.gnn_model import STGNNModel
from utils.data_loader import create_sequence_dataset, graph_to_edge_index
from graph_builder import build_kerala_district_graph


def load_data(seq_len: int = 5):
    """Load graph and sequence dataset with shapes expected by STGNN."""
    data_path = PROJECT_ROOT / "data" / "clean_flood_data.csv"
    df = pd.read_csv(data_path)

    graph = build_kerala_district_graph()
    edge_index, district_order = graph_to_edge_index(graph)

    missing_cols = [district for district in district_order if district not in df.columns]
    if missing_cols:
        raise ValueError(
            "Missing district columns in clean_flood_data.csv: " + ", ".join(missing_cols)
        )

    # Align district columns to graph node order so node indices remain consistent.
    district_data = df[district_order]
    x, y = create_sequence_dataset(district_data.values, seq_len=seq_len)

    if not torch.is_tensor(edge_index):
        edge_index = torch.tensor(edge_index, dtype=torch.long)

    return x.float(), y.float(), edge_index.long()


def train_model():
    x, y, edge_index = load_data(seq_len=5)

    # Split: 80% train, 20% test.
    num_samples = x.size(0)
    train_size = int(0.8 * num_samples)
    test_size = num_samples - train_size

    permutation = torch.randperm(num_samples)
    train_idx = permutation[:train_size]
    test_idx = permutation[train_size:]

    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    print(f"Train samples: {x_train.size(0)}, Test samples: {x_test.size(0)}")
    print(f"X shape: {tuple(x.shape)}, y shape: {tuple(y.shape)}, edge_index shape: {tuple(edge_index.shape)}")

    model = STGNNModel(spatial_hidden_dim=16, temporal_hidden_dim=32)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 100
    train_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        predictions = model(x_train, edge_index)
        loss = criterion(predictions, y_train)

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}/{epochs} - Train Loss: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        test_predictions = model(x_test, edge_index)
        test_loss = criterion(test_predictions, y_test).item()
    print(f"Final Test Loss: {test_loss:.6f}")

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    model_path = results_dir / "trained_model.pth"
    torch.save(model.state_dict(), model_path)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), train_losses, color="teal", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("STGNN Training Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    loss_curve_path = results_dir / "loss_curve.png"
    plt.savefig(loss_curve_path, dpi=150)
    plt.close()

    print(f"Model saved to: {model_path}")
    print(f"Loss curve saved to: {loss_curve_path}")
    print("Training complete!")


if __name__ == "__main__":
    train_model()
