from pathlib import Path

import pandas as pd


def print_dataset_overview(name: str, df: pd.DataFrame) -> None:
	print(f"\n{'=' * 70}")
	print(f"Dataset: {name}")
	print(f"{'=' * 70}")

	print("\n1) Shape:")
	print(df.shape)

	print("\n2) Columns:")
	print(df.columns.tolist())

	print("\n3) First 5 rows:")
	print(df.head())

	print("\n4) Missing values (per column):")
	print(df.isnull().sum())

	print("\n5) Basic statistics (mean, min, max):")
	numeric_cols = df.select_dtypes(include="number")
	if numeric_cols.empty:
		print("No numeric columns found.")
	else:
		stats = numeric_cols.agg(["mean", "min", "max"]) 
		print(stats)


def main() -> None:
	project_root = Path(__file__).resolve().parents[1]
	data_dir = project_root / "data"

	district_file = data_dir / "district_wise_details.csv"
	warning_file = data_dir / "warnings_actual_predicted.csv"

	district_df = pd.read_csv(district_file)
	warning_df = pd.read_csv(warning_file)

	print_dataset_overview("district_wise_details.csv", district_df)
	print_dataset_overview("warnings_actual_predicted.csv", warning_df)


if __name__ == "__main__":
	main()
