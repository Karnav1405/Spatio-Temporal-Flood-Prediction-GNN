import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class STGNNModel(nn.Module):
    """Spatio-Temporal GNN for district-level flood prediction.

    Input shape:
        x: (batch_size, time_steps, num_nodes)

    Output shape:
        predictions: (batch_size, num_nodes)
    """

    def __init__(self, spatial_hidden_dim: int = 16, temporal_hidden_dim: int = 32):
        super().__init__()

        # Spatial layer (GCN): learns how each district is influenced by connected districts
        # at each time step. Each node starts with 1 feature (flood value for that day).
        self.gcn = GCNConv(in_channels=1, out_channels=spatial_hidden_dim)

        # Temporal layer (GRU): learns how each district's spatial embedding evolves
        # across multiple days.
        self.gru = nn.GRU(
            input_size=spatial_hidden_dim,
            hidden_size=temporal_hidden_dim,
            batch_first=True,
        )

        # Final prediction layer: maps temporal representation to next-day flood value
        # for each node (district).
        self.fc = nn.Linear(temporal_hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features over time, shape (batch, time_steps, num_nodes)
            edge_index: Graph connectivity, shape (2, num_edges)

        Returns:
            Tensor of shape (batch, num_nodes) with next-day predictions per node.
        """
        batch_size, time_steps, num_nodes = x.shape

        spatial_outputs = []
        for t in range(time_steps):
            # Features for one day: (batch, num_nodes) -> (batch, num_nodes, 1)
            x_t = x[:, t, :].unsqueeze(-1)

            # Apply GCN for each sample in the batch at this time step.
            gcn_batch = []
            for b in range(batch_size):
                # For one graph sample: (num_nodes, 1) -> (num_nodes, spatial_hidden_dim)
                node_embeddings = self.gcn(x_t[b], edge_index)
                gcn_batch.append(node_embeddings)

            # Stack back into batch form: (batch, num_nodes, spatial_hidden_dim)
            gcn_batch = torch.stack(gcn_batch, dim=0)
            spatial_outputs.append(gcn_batch)

        # Combine all time steps:
        # (time_steps, batch, num_nodes, spatial_hidden_dim) ->
        # (batch, time_steps, num_nodes, spatial_hidden_dim)
        spatial_seq = torch.stack(spatial_outputs, dim=1)

        # Prepare for GRU by treating each node as a sequence across time:
        # (batch, time_steps, num_nodes, spatial_hidden_dim) ->
        # (batch * num_nodes, time_steps, spatial_hidden_dim)
        gru_input = spatial_seq.permute(0, 2, 1, 3).contiguous().view(
            batch_size * num_nodes, time_steps, -1
        )

        # GRU output over time for each node sequence.
        gru_out, _ = self.gru(gru_input)

        # Use last time step output as summary representation.
        last_hidden = gru_out[:, -1, :]  # (batch * num_nodes, temporal_hidden_dim)

        # Predict one value per node sequence and reshape back to (batch, num_nodes).
        preds = self.fc(last_hidden).view(batch_size, num_nodes)
        return preds


if __name__ == "__main__":
    # Random test input matching your project shapes.
    x_test = torch.randn(10, 5, 14)  # (samples, days, districts)

    # Random edge index with shape (2, 48), matching your graph edge count.
    edge_index_test = torch.randint(0, 14, (2, 48), dtype=torch.long)

    model = STGNNModel(spatial_hidden_dim=16, temporal_hidden_dim=32)
    y_pred = model(x_test, edge_index_test)

    print("Output shape:", y_pred.shape)

    if y_pred.shape == (10, 14):
        print("Model architecture ready!")
    else:
        raise RuntimeError(
            f"Unexpected output shape: {tuple(y_pred.shape)}. Expected (10, 14)."
        )
