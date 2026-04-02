from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


WARNING_TO_SEVERITY = {
    "Green": 0,
    "Yellow": 1,
    "Orange": 2,
    "Red": 3,
}


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_file = project_root / "data" / "warnings_actual_predicted.csv"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_file)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["actual_severity"] = df["actual_rainfall"].map(WARNING_TO_SEVERITY)
    df["predicted_severity"] = df["predicted_rainfall"].map(WARNING_TO_SEVERITY)

    missing_actual = df["actual_severity"].isna().sum()
    missing_predicted = df["predicted_severity"].isna().sum()
    if missing_actual > 0 or missing_predicted > 0:
        print("Warning: Unmapped warning categories found.")
        print(f"Unmapped actual warnings: {missing_actual}")
        print(f"Unmapped predicted warnings: {missing_predicted}")

    # Heatmap data: district vs date using actual warning severity.
    heatmap_data = (
        df.pivot_table(
            index="district",
            columns="date",
            values="actual_severity",
            aggfunc="mean",
        )
        .sort_index()
        .sort_index(axis=1)
    )

    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(heatmap_data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=3)
    ax.set_title("Kerala Flood Severity Over Time (Actual Warnings)")
    ax.set_xlabel("Date")
    ax.set_ylabel("District")

    xticks = range(len(heatmap_data.columns))
    ax.set_xticks(xticks)
    ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in heatmap_data.columns], rotation=90)
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index)

    colorbar = fig.colorbar(im, ax=ax)
    colorbar.set_label("Severity (Green=0, Yellow=1, Orange=2, Red=3)")

    heatmap_path = results_dir / "district_severity_heatmap.png"
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=300)
    plt.close(fig)

    actual_severe = df[df["actual_severity"] >= 2].groupby("district").size()
    predicted_severe = df[df["predicted_severity"] >= 2].groupby("district").size()

    severe_counts = pd.DataFrame(
        {
            "Actual Red/Orange": actual_severe,
            "Predicted Red/Orange": predicted_severe,
        }
    ).fillna(0)

    severe_counts["Total Severe Warnings"] = (
        severe_counts["Actual Red/Orange"] + severe_counts["Predicted Red/Orange"]
    )
    severe_counts = severe_counts.sort_values("Total Severe Warnings", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    severe_counts[["Actual Red/Orange", "Predicted Red/Orange"]].plot(
        kind="bar",
        ax=ax,
        color=["#E76F51", "#2A9D8F"],
    )
    ax.set_title("Districts with Most Red/Orange Warnings")
    ax.set_xlabel("District")
    ax.set_ylabel("Count of Red/Orange Warnings")
    ax.legend(title="Warning Type")
    ax.tick_params(axis="x", rotation=45)

    bar_chart_path = results_dir / "district_red_orange_warnings.png"
    plt.tight_layout()
    plt.savefig(bar_chart_path, dpi=300)
    plt.close(fig)

    print(f"Saved heatmap to: {heatmap_path}")
    print(f"Saved bar chart to: {bar_chart_path}")


if __name__ == "__main__":
    main()
