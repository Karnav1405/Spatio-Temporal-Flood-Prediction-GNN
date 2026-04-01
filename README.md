# Spatio-Temporal Flood Prediction Using Graph Neural Networks

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Progress-F39C12)

## Project Description

This college project focuses on forecasting flood risk by modeling both spatial relationships between geographic locations and temporal changes in hydrological patterns. We use a spatio-temporal deep learning pipeline that combines Graph Neural Networks (GNNs) for location connectivity with recurrent temporal modeling to capture evolving flood dynamics over time. The goal is to build an interpretable and scalable framework for early flood prediction support.

## How It Works

- Spatial modeling with GNN: each node represents a location (for example, sensor station or region), and edges represent geographic or hydrological connectivity.
- Node features: each node carries flood-related features such as rainfall, water level, and other environmental variables.
- Graph convolution for spatial context: a GNN layer aggregates information from neighboring nodes to learn spatial dependencies.
- Temporal modeling with GRU: sequential node embeddings are passed through a GRU to learn how flood behavior changes over time.
- Prediction head: the model outputs flood risk or flood level predictions for future time steps.

## Project Structure

```text
flood-gnn-project/
|-- data/         # Raw and processed datasets
|-- notebooks/    # EDA, experiments, and training notebooks
|-- models/       # Model architectures and training scripts
|-- results/      # Outputs: metrics, plots, and saved artifacts
|-- utils/        # Helper functions and utility modules
|-- test_setup.py # Environment and dependency validation script
```

## Libraries Used

| Library | Version |
|---|---|
| torch | 2.11.0 |
| torch-geometric | 2.7.0 |
| pandas | 2.3.1 |
| numpy | 2.3.2 |
| matplotlib | 3.10.3 |
| networkx | 3.6.1 |
| scikit-learn | 1.8.0 |

## How To Run Locally

1. Clone the repository:

	```bash
	git clone <your-repo-url>
	cd flood-gnn-project
	```

2. Create and activate a virtual environment:

	```bash
	python -m venv .venv
	# Windows PowerShell
	.venv\Scripts\Activate.ps1
	```

3. Install dependencies:

	```bash
	pip install torch==2.11.0 torch-geometric==2.7.0 pandas==2.3.1 numpy==2.3.2 matplotlib==3.10.3 networkx==3.6.1 scikit-learn==1.8.0
	```

4. Verify setup:

	```bash
	python test_setup.py
	```

5. Start experimentation in the notebooks folder and move reusable code into models and utils.

## Notes

This project is currently in active development. Features, training pipelines, and evaluation protocols will be refined incrementally each week.
