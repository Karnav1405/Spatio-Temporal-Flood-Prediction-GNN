import traceback

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


def print_versions() -> None:
	versions = {
		"torch": torch.__version__,
		"torch_geometric": torch_geometric.__version__,
		"pandas": pd.__version__,
		"numpy": np.__version__,
		"matplotlib": matplotlib.__version__,
		"networkx": nx.__version__,
	}

	print("Library versions:")
	for lib_name, lib_version in versions.items():
		print(f"- {lib_name}: {lib_version}")


def run_tiny_gcn_test() -> None:
	# 5 nodes, 3 input features per node.
	x = torch.randn((5, 3), dtype=torch.float)

	# Undirected chain-like graph encoded with bidirectional edges.
	edge_index = torch.tensor(
		[
			[0, 1, 1, 2, 2, 3, 3, 4],
			[1, 0, 2, 1, 3, 2, 4, 3],
		],
		dtype=torch.long,
	)

	data = Data(x=x, edge_index=edge_index)
	conv = GCNConv(in_channels=3, out_channels=2)

	out = conv(data.x, data.edge_index)
	print("GCN output shape:", tuple(out.shape))


def main() -> None:
	print_versions()
	run_tiny_gcn_test()
	print("Setup successful!")


if __name__ == "__main__":
	try:
		main()
	except Exception:
		print("Setup failed. See traceback below:")
		traceback.print_exc()
