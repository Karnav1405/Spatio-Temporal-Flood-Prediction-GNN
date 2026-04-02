from pathlib import Path

import pandas as pd


WARNING_TO_SEVERITY = {
    "Green": 0,
    "Yellow": 1,
    "Orange": 2,
    "Red": 3,
}


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    input_file = project_root / "data" / "warnings_actual_predicted.csv"
    output_file = project_root / "data" / "clean_flood_data.csv"

    df = pd.read_csv(input_file)

    df["actual_severity"] = df["actual_rainfall"].map(WARNING_TO_SEVERITY)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    clean_pivot = (
        df.pivot_table(
            index="date",
            columns="district",
            values="actual_severity",
            aggfunc="mean",
        )
        .sort_index()
        .sort_index(axis=1)
    )

    clean_pivot.to_csv(output_file)
    print("Clean data saved!")


if __name__ == "__main__":
    main()
