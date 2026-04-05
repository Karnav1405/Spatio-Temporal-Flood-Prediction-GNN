from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st
import torch


# Make project and utils importable when running with Streamlit.
PROJECT_ROOT = Path(__file__).resolve().parent
UTILS_DIR = PROJECT_ROOT / "utils"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from models.gnn_model import STGNNModel
from utils.data_loader import graph_to_edge_index
from graph_builder import build_kerala_district_graph


WARNING_LABELS = {0: "Green", 1: "Yellow", 2: "Orange", 3: "Red"}
WARNING_COLORS = {
    0: "#2E8B57",  # Green
    1: "#E6B800",  # Yellow
    2: "#FF8C00",  # Orange
    3: "#D62828",  # Red
}
WARNING_STATUS = {0: "Safe", 1: "Watch", 2: "Warning", 3: "Danger"}
LEVEL_BG_COLORS = {
    0: "#E8F5E9",
    1: "#FFFDE7",
    2: "#FFF3E0",
    3: "#FFEBEE",
}

MODEL_PATH = PROJECT_ROOT / "results" / "trained_model.pth"
DATA_PATH = PROJECT_ROOT / "data" / "clean_flood_data.csv"
GRAPH_IMG_PATH = PROJECT_ROOT / "results" / "kerala_district_graph.png"


@st.cache_data(show_spinner=False)
def load_dataset_and_graph():
    """Load flood data, graph, edge index, and district ordering."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing data file: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    graph = build_kerala_district_graph()
    edge_index, district_order = graph_to_edge_index(graph)

    missing_cols = [district for district in district_order if district not in df.columns]
    if missing_cols:
        raise ValueError(
            "Missing district columns in clean_flood_data.csv: " + ", ".join(missing_cols)
        )

    district_data = df[district_order].astype(np.float32)

    if not torch.is_tensor(edge_index):
        edge_index = torch.tensor(edge_index, dtype=torch.long)

    return district_data, graph, edge_index.long(), district_order


@st.cache_resource(show_spinner=False)
def load_model():
    """Load trained ST-GNN model weights once per app session."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing trained model: {MODEL_PATH}")

    model = STGNNModel(spatial_hidden_dim=16, temporal_hidden_dim=32)
    state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def warning_badge(level: int) -> str:
    """Return colored HTML badge for warning level."""
    label = WARNING_LABELS[level]
    color = WARNING_COLORS[level]
    return (
        f"<span style='display:inline-block; margin:4px; padding:6px 12px; "
        f"border-radius:999px; background:{color}; color:white; font-weight:700;'>"
        f"{label}</span>"
    )


def warning_legend_box(level: int) -> str:
    """Return small colored legend box for sidebar."""
    label = WARNING_LABELS[level]
    status = WARNING_STATUS[level].lower()
    color = WARNING_COLORS[level]
    return (
        "<div style='display:flex; align-items:center; gap:8px; margin:6px 0;'>"
        f"<span style='display:inline-block; width:14px; height:14px; border-radius:4px; background:{color};'></span>"
        f"<span style='color:#2C3E50; font-size:13px;'><strong>{label}</strong> = {status}</span>"
        "</div>"
    )


def run_prediction(selected_district: str, day_levels, district_data, edge_index, district_order):
    """Run one-step ST-GNN prediction for all districts."""
    if len(district_data) < 5:
        raise ValueError("At least 5 days of data are required for prediction.")

    # Start from latest 5-day state for all districts, then inject user-input sequence
    # for the selected district.
    recent_window = district_data.tail(5).to_numpy().copy()  # (5, 14)
    district_idx = district_order.index(selected_district)
    recent_window[:, district_idx] = np.array(day_levels, dtype=np.float32)

    x_input = torch.tensor(recent_window, dtype=torch.float32).unsqueeze(0)  # (1, 5, 14)

    model = load_model()
    with torch.no_grad():
        y_pred = model(x_input, edge_index).squeeze(0).cpu().numpy()

    y_pred_levels = np.clip(np.rint(y_pred), 0, 3).astype(int)
    return y_pred, y_pred_levels


def main():
    st.set_page_config(page_title="Kerala Flood Prediction — ST-GNN", layout="wide")

    st.markdown(
        """
        <style>
            .stApp {
                background: #FFFFFF;
                color: #2C3E50;
            }

            [data-testid="stSidebar"] {
                background: #F8F9FA;
                border-right: 1px solid #E5E7EB;
            }

            [data-testid="stSidebar"] * {
                color: #2C3E50 !important;
            }

            .hero {
                background: #FFFFFF;
                border-top: 4px solid #0097A7;
                border-radius: 12px;
                padding: 22px 24px 16px 24px;
                border-left: 1px solid #E5E7EB;
                border-right: 1px solid #E5E7EB;
                border-bottom: 1px solid #E5E7EB;
                margin-bottom: 18px;
            }

            .hero-title {
                margin: 0;
                color: #2C3E50;
                font-size: 38px;
                font-weight: 800;
                line-height: 1.15;
            }

            .hero-subtitle {
                color: #2C3E50;
                opacity: 0.85;
                margin-top: 8px;
                margin-bottom: 12px;
                font-size: 16px;
            }

            .hero-divider {
                border: none;
                border-top: 2px solid #0097A7;
                margin: 0;
            }

            .metric-card {
                background: #FFFFFF;
                border: 1px solid #E5E7EB;
                border-top: 4px solid #0097A7;
                border-radius: 12px;
                padding: 16px;
                min-height: 110px;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
            }

            .metric-value {
                color: #0097A7;
                font-size: 30px;
                font-weight: 800;
                line-height: 1;
                margin-bottom: 8px;
            }

            .metric-label {
                color: #2C3E50;
                font-size: 13px;
                font-weight: 600;
                letter-spacing: 0.2px;
            }

            .content-card {
                background: #FFFFFF;
                border: 1px solid #E5E7EB;
                border-radius: 12px;
                padding: 18px;
                margin-bottom: 16px;
            }

            .result-card {
                border-radius: 12px;
                padding: 20px;
                border-left: 10px solid #0097A7;
                margin: 10px 0 12px 0;
            }

            .result-title {
                font-size: 28px;
                font-weight: 800;
                margin-bottom: 6px;
            }

            .result-subtitle {
                font-size: 14px;
                color: #2C3E50;
                opacity: 0.9;
            }

            .neighbor-card {
                background: #FFFFFF;
                border: 1px solid #E5E7EB;
                border-radius: 10px;
                padding: 12px;
                margin-bottom: 10px;
                box-shadow: 0 1px 1px rgba(0, 0, 0, 0.03);
            }

            .graph-card {
                background: #FFFFFF;
                border: 1px solid #E5E7EB;
                border-radius: 12px;
                padding: 18px;
                margin-top: 10px;
            }

            .graph-description {
                color: #2C3E50;
                opacity: 0.85;
                margin-top: 6px;
                margin-bottom: 14px;
                font-size: 14px;
            }

            .graph-image-wrap {
                border: 1px solid #DDE3E8;
                border-radius: 10px;
                padding: 10px;
                background: #FFFFFF;
            }

            div[data-testid="stSidebar"] button[kind="primary"] {
                width: 100%;
                background: #0097A7;
                border: 1px solid #0097A7;
                color: #FFFFFF;
                font-weight: 700;
                border-radius: 8px;
            }

            div[data-testid="stSidebar"] button[kind="primary"]:hover {
                background: #008391;
                border-color: #008391;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class='hero'>
            <h1 class='hero-title'>Kerala Flood Prediction</h1>
            <div class='hero-subtitle'>Spatio-Temporal Graph Neural Network — Kerala Floods 2018 Case Study</div>
            <hr class='hero-divider' />
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4)
    metric_html = [
        ("82.14%", "Model Accuracy"),
        ("14", "Districts Monitored"),
        ("40", "Training Samples"),
        ("ST-GNN", "Model Type"),
    ]
    for col, (value, label) in zip(metric_cols, metric_html):
        col.markdown(
            (
                "<div class='metric-card'>"
                f"<div class='metric-value'>{value}</div>"
                f"<div class='metric-label'>{label}</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

    try:
        district_data, graph, edge_index, district_order = load_dataset_and_graph()
        _ = load_model()
    except Exception as exc:
        st.error(f"Startup error: {exc}")
        st.stop()

    st.sidebar.header("Input Controls")
    selected_district = st.sidebar.selectbox("Select Kerala district", district_order)

    st.sidebar.markdown("### Enter last 5 days warning levels")
    slider_levels = []
    for i in range(5):
        level = st.sidebar.slider(
            f"Day {i + 1} warning level",
            min_value=0,
            max_value=3,
            value=1,
            help="0=Green, 1=Yellow, 2=Orange, 3=Red",
        )
        slider_levels.append(level)

    st.sidebar.markdown("### Warning Level Legend")
    st.sidebar.markdown(
        "".join(warning_legend_box(level) for level in [0, 1, 2, 3]),
        unsafe_allow_html=True,
    )

    predict_clicked = st.sidebar.button("Predict Tomorrow", type="primary")

    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.subheader("Selected District")
    st.write(f"**{selected_district}**")

    st.subheader("Input Summary (Last 5 Days)")
    badges_html = "".join(warning_badge(level) for level in slider_levels)
    st.markdown(badges_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if predict_clicked:
        try:
            y_pred_raw, y_pred_levels = run_prediction(
                selected_district=selected_district,
                day_levels=slider_levels,
                district_data=district_data,
                edge_index=edge_index,
                district_order=district_order,
            )

            district_idx = district_order.index(selected_district)
            district_level = int(y_pred_levels[district_idx])
            district_label = WARNING_LABELS[district_level]
            district_status = WARNING_STATUS[district_level]
            district_color = WARNING_COLORS[district_level]
            district_bg = LEVEL_BG_COLORS[district_level]

            st.subheader("Tomorrow Prediction")
            st.markdown(
                (
                    f"<div class='result-card' style='background:{district_bg}; border-left-color:{district_color};'>"
                    f"<div class='result-title' style='color:{district_color};'>{district_label} - {district_status}</div>"
                    f"<div class='result-subtitle'>Model score: {y_pred_raw[district_idx]:.3f}</div>"
                    f"</div>"
                ),
                unsafe_allow_html=True,
            )

            st.subheader("Neighboring District Predictions")
            neighbors = sorted(list(graph.neighbors(selected_district)))

            neighbor_rows = []
            for name in neighbors:
                idx = district_order.index(name)
                level = int(y_pred_levels[idx])
                neighbor_rows.append(
                    {
                        "District": name,
                        "Level": WARNING_LABELS[level],
                        "Status": WARNING_STATUS[level],
                        "Score": round(float(y_pred_raw[idx]), 3),
                    }
                )

            if neighbor_rows:
                neighbor_cols = st.columns(2)
                for idx, row in enumerate(neighbor_rows):
                    lvl = [k for k, v in WARNING_LABELS.items() if v == row["Level"]][0]
                    bg = LEVEL_BG_COLORS[lvl]
                    color = WARNING_COLORS[lvl]
                    neighbor_cols[idx % 2].markdown(
                        (
                            f"<div class='neighbor-card' style='border-left:6px solid {color}; background:{bg};'>"
                            f"<div style='font-weight:700; color:#2C3E50;'>{row['District']}</div>"
                            f"<div style='margin-top:4px; color:{color}; font-weight:700;'>{row['Level']} - {row['Status']}</div>"
                            f"<div style='font-size:13px; color:#2C3E50; margin-top:3px;'>Score: {row['Score']}</div>"
                            "</div>"
                        ),
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No neighboring districts found for this node.")

        except Exception as exc:
            st.error(f"Prediction error: {exc}")

    st.markdown("<div class='graph-card'>", unsafe_allow_html=True)
    st.subheader("Kerala District Graph")
    st.markdown(
        "<div class='graph-description'>Geographic connections between 14 Kerala districts used for spatial learning</div>",
        unsafe_allow_html=True,
    )
    if GRAPH_IMG_PATH.exists():
        st.markdown("<div class='graph-image-wrap'>", unsafe_allow_html=True)
        st.image(str(GRAPH_IMG_PATH), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning(
            "Graph visualization not found at results/kerala_district_graph.png. "
            "Run utils/graph_builder.py to generate it."
        )
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
