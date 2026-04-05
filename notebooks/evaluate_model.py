from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch

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
    """Load graph, sequence tensors, and district names."""
    data_path = PROJECT_ROOT / "data" / "clean_flood_data.csv"
    df = pd.read_csv(data_path)

    graph = build_kerala_district_graph()
    edge_index, district_order = graph_to_edge_index(graph)

    missing_cols = [district for district in district_order if district not in df.columns]
    if missing_cols:
        raise ValueError(
            "Missing district columns in clean_flood_data.csv: " + ", ".join(missing_cols)
        )

    district_data = df[district_order]
    x, y = create_sequence_dataset(district_data.values, seq_len=seq_len)

    if not torch.is_tensor(edge_index):
        edge_index = torch.tensor(edge_index, dtype=torch.long)

    return x.float(), y.float(), edge_index.long(), district_order


def split_data(x: torch.Tensor, y: torch.Tensor, train_ratio: float = 0.8, seed: int = 42):
    """Create reproducible 80/20 train-test split."""
    num_samples = x.size(0)
    train_size = int(train_ratio * num_samples)

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(num_samples, generator=generator)

    train_idx = permutation[:train_size]
    test_idx = permutation[train_size:]

    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


def compute_metrics(pred: torch.Tensor, target: torch.Tensor):
    """Compute MAE, RMSE, and warning-level accuracy."""
    mae = torch.mean(torch.abs(pred - target)).item()
    rmse = torch.sqrt(torch.mean((pred - target) ** 2)).item()

    # Accuracy on warning levels by rounding predictions to nearest class-like value.
    pred_labels = torch.round(pred)
    target_labels = torch.round(target)
    accuracy = (pred_labels == target_labels).float().mean().item() * 100.0

    return mae, rmse, accuracy


def plot_predictions_vs_actual(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    district_names,
    save_path: Path,
):
    """Plot predicted vs actual warning levels for each district."""
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    fig, axes = plt.subplots(7, 2, figsize=(14, 18), sharex=True)
    axes = axes.flatten()

    x_axis = range(1, y_true_np.shape[0] + 1)

    for idx, district in enumerate(district_names):
        ax = axes[idx]
        ax.plot(x_axis, y_true_np[:, idx], marker="o", linewidth=2, label="Actual")
        ax.plot(x_axis, y_pred_np[:, idx], marker="x", linewidth=2, linestyle="--", label="Predicted")
        ax.set_title(district)
        ax.set_ylabel("Warning Level")
        ax.grid(alpha=0.3)

    for idx in range(len(district_names), len(axes)):
        axes[idx].axis("off")

    axes[-1].set_xlabel("Test Sample Index")
    axes[-2].set_xlabel("Test Sample Index")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)

    fig.suptitle("Predicted vs Actual Warning Levels by District", fontsize=14, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    x, y, edge_index, district_names = load_data(seq_len=5)
    _, _, x_test, y_test = split_data(x, y, train_ratio=0.8, seed=42)

    model_path = PROJECT_ROOT / "results" / "trained_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at: {model_path}")

    model = STGNNModel(spatial_hidden_dim=16, temporal_hidden_dim=32)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    with torch.no_grad():
        y_pred = model(x_test, edge_index)

    model_mae, model_rmse, model_acc = compute_metrics(y_pred, y_test)

    # Baseline: predict tomorrow equals yesterday (last observed day in input window).
    baseline_pred = x_test[:, -1, :]
    baseline_mae, baseline_rmse, baseline_acc = compute_metrics(baseline_pred, y_test)

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_path = results_dir / "predictions_vs_actual.png"
    plot_predictions_vs_actual(y_test, y_pred, district_names, plot_path)

    print("Evaluation Metrics (Model)")
    print(f"- MAE: {model_mae:.6f}")
    print(f"- RMSE: {model_rmse:.6f}")
    print(f"- Accuracy: {model_acc:.2f}%")

    print("\nBaseline Metrics (Yesterday = Tomorrow)")
    print(f"- MAE: {baseline_mae:.6f}")
    print(f"- RMSE: {baseline_rmse:.6f}")
    print(f"- Accuracy: {baseline_acc:.2f}%")

    print("\nMAE Comparison")
    print(f"- Model MAE: {model_mae:.6f}")
    print(f"- Baseline MAE: {baseline_mae:.6f}")

    print(f"\nPrediction plot saved to: {plot_path}")
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
