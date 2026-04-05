from pathlib import Path
import sys

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
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
    """Load sequence data and graph edge index."""
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
    """Reproducible train-test split used for final analysis."""
    num_samples = x.size(0)
    train_size = int(train_ratio * num_samples)

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(num_samples, generator=generator)

    test_idx = permutation[train_size:]
    return x[test_idx], y[test_idx]


def get_predictions():
    """Load trained model and produce predictions on test split."""
    x, y, edge_index, district_names = load_data(seq_len=5)
    x_test, y_test = split_data(x, y, train_ratio=0.8, seed=42)

    model_path = PROJECT_ROOT / "results" / "trained_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = STGNNModel(spatial_hidden_dim=16, temporal_hidden_dim=32)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    with torch.no_grad():
        y_pred = model(x_test, edge_index)

    return y_test.cpu().numpy(), y_pred.cpu().numpy(), district_names


def find_best_worst_district(y_true: np.ndarray, y_pred: np.ndarray):
    """Return indices of best and worst districts by MAE."""
    district_mae = np.mean(np.abs(y_pred - y_true), axis=0)
    best_idx = int(np.argmin(district_mae))
    worst_idx = int(np.argmax(district_mae))
    return best_idx, worst_idx, district_mae


def build_final_summary_plot():
    """Create the final 4-panel summary figure."""
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    loss_curve_path = results_dir / "loss_curve.png"
    if not loss_curve_path.exists():
        raise FileNotFoundError(f"Loss curve image not found: {loss_curve_path}")

    y_true, y_pred, district_names = get_predictions()
    best_idx, worst_idx, district_mae = find_best_worst_district(y_true, y_pred)

    # Metrics from final reported results.
    metric_names = ["Accuracy (%)", "RMSE", "MAE"]
    model_metrics = [82.14, 0.778, 0.464]
    baseline_metrics = [78.57, 0.963, 0.428]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1) Training loss curve image.
    ax1 = axes[0, 0]
    loss_img = mpimg.imread(loss_curve_path)
    ax1.imshow(loss_img)
    ax1.axis("off")
    ax1.set_title("Training Loss Curve")

    # 2) Model vs baseline bar chart.
    ax2 = axes[0, 1]
    x_pos = np.arange(len(metric_names))
    width = 0.35
    ax2.bar(x_pos - width / 2, model_metrics, width=width, label="ST-GNN", color="#2C7BB6")
    ax2.bar(x_pos + width / 2, baseline_metrics, width=width, label="Baseline", color="#D7191C")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(metric_names)
    ax2.set_title("Model vs Baseline")
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend()

    # 3) Predicted vs actual for best district.
    ax3 = axes[1, 0]
    sample_axis = np.arange(1, y_true.shape[0] + 1)
    ax3.plot(sample_axis, y_true[:, best_idx], marker="o", linewidth=2, label="Actual")
    ax3.plot(sample_axis, y_pred[:, best_idx], marker="x", linewidth=2, linestyle="--", label="Predicted")
    ax3.set_title(f"Best District: {district_names[best_idx]} (MAE={district_mae[best_idx]:.3f})")
    ax3.set_xlabel("Test Sample")
    ax3.set_ylabel("Warning Level")
    ax3.grid(alpha=0.3)
    ax3.legend()

    # 4) Predicted vs actual for worst district.
    ax4 = axes[1, 1]
    ax4.plot(sample_axis, y_true[:, worst_idx], marker="o", linewidth=2, label="Actual")
    ax4.plot(sample_axis, y_pred[:, worst_idx], marker="x", linewidth=2, linestyle="--", label="Predicted")
    ax4.set_title(f"Worst District: {district_names[worst_idx]} (MAE={district_mae[worst_idx]:.3f})")
    ax4.set_xlabel("Test Sample")
    ax4.set_ylabel("Warning Level")
    ax4.grid(alpha=0.3)
    ax4.legend()

    fig.suptitle("Kerala Flood Prediction: Final ST-GNN Summary", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = results_dir / "final_summary.png"
    plt.savefig(output_path, dpi=180)
    plt.close(fig)

    return output_path


def main():
    output_path = build_final_summary_plot()
    print(f"Saved summary to: {output_path}")
    print("Final results saved!")


if __name__ == "__main__":
    main()
