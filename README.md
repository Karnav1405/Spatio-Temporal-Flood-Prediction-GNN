# Spatio-Temporal Flood Prediction Using Graph Neural Networks

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/Status-Data%20Exploration%20Complete-27AE60)

## Project Description

This college project focuses on forecasting flood risk by modeling both spatial relationships between geographic locations and temporal changes in hydrological patterns. We use a spatio-temporal deep learning pipeline that combines Graph Neural Networks (GNNs) for location connectivity with recurrent temporal modeling to capture evolving flood dynamics over time. The goal is to build an interpretable and scalable framework for early flood prediction support.

This project applies spatio-temporal deep learning to the 2018 Kerala floods as a real-world case study for early flood warning prediction.

## Dataset

- Dataset: Kerala Floods 2018
- Graph nodes: 14 Kerala districts treated as graph nodes
- Temporal coverage: daily flood warning levels (Green/Yellow/Orange/Red) for August 2018
- Key finding: Idukki and Wayanad were the most affected districts, with peak flood severity during August 15–17
- Data source: Kaggle (Devakumar K.P.)

## Visualizations

- Kerala District Graph: 14 districts as nodes, 24 edges representing shared borders between neighboring districts. High risk districts (Idukki, Wayanad, Malappuram) highlighted in red.
- Peak Flood Day Map (August 15, 2018): Every district shown in red confirming August 15 was a state-wide catastrophe with no district safe.

## Graph Details

- Nodes: 14 Kerala districts
- Edges: 24 border connections (48 directional)
- Input to model: 5 days of flood severity across all 14 districts
- Output from model: next day warning level for all 14 districts

## Model Architecture

- Model name: ST-GNN (Spatio-Temporal Graph Neural Network)
- Layer 1: GCNConv - spatial learning across neighboring districts
- Layer 2: GRU - temporal learning across 5 days of history
- Layer 3: Linear - outputs flood warning prediction
- Input: 5 days of flood severity across 14 districts
- Output: next day warning level for all 14 districts

## Training Results

- Training samples: 8, Test samples: 2
- Optimizer: Adam, Loss function: MSE
- Epochs: 100
- Initial loss: 1.869, Final training loss: 0.582
- Test loss: 0.863
- Loss improved by 69% during training

## How It Works

- Spatial modeling with GNN: each node represents a Kerala district, and edges represent geographic connectivity between neighboring districts
- Node features: each node carries flood-related features such as daily warning levels and rainfall severity
- Graph convolution for spatial context: a GNN layer aggregates information from neighboring districts to learn spatial flood spread patterns
- Temporal modeling with GRU: sequential node embeddings are passed through a GRU to learn how flood warning levels change over time
- Prediction head: the model outputs flood warning level predictions (Green/Yellow/Orange/Red) for future time steps

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
	git clone https://github.com/Karnav1405/Spatio-Temporal-Flood-Prediction-GNN.git
	cd flood-gnn-project
```

2. Create and activate a virtual environment:
```bash
	python -m venv .venv
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