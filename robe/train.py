# train_stgnn_stream.py
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv
from torch_geometric_temporal.nn.recurrent import GConvGRU
from tqdm import tqdm
import networkx as nx
import numpy as np

# ---- Hyperparams ----
NUM_NODES = 200 * 200
NODE_FEAT_DIM = 3            # [count, sin_t, cos_t]
HIDDEN_DIM = 64
OUT_DIM = 1                  # predict next-step user count (regression). For classification set >1.
SEQ_LEN = 8                  # number of input timesteps to the recurrent model
LR = 1e-3
EPOCHS = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Model: GConvGRU-based (stacked) ----
class STGNN(torch.nn.Module):
    def __init__(self, node_feat_dim, hidden_dim, out_dim):
        super().__init__()
        # recurrent spatial conv unit
        self.recurrent = GConvGRU(in_channels=node_feat_dim, out_channels=hidden_dim, K=2)
        # readout conv layers
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.lin = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        # x: [num_nodes, node_feat_dim]
        # edge_index: torch.long tensor [2, num_edges]
        h = self.recurrent(x, edge_index)  # returns [num_nodes, hidden_dim]
        h = F.relu(self.conv1(h, edge_index))
        out = self.lin(h)  # [num_nodes, out_dim]
        return out.squeeze(-1)  # [num_nodes] for out_dim==1

# ---- Utility: networkx snapshot -> torch_geometric Data object ----
def nx_to_pyg_data(G):
    """
    Expects G to have nodes 0..NUM_NODES-1 and node attribute 'feat' (list/np array length NODE_FEAT_DIM).
    Also expects edges present between nodes (spatial + temporal edges for that snapshot).
    """
    # use torch_geometric.utils.from_networkx but create proper node feature matrix
    # Build feature matrix [num_nodes, feat_dim]
    feat = np.zeros((NUM_NODES, NODE_FEAT_DIM), dtype=np.float32)
    for n, d in G.nodes(data=True):
        if "feat" in d:
            feat[n] = np.array(d["feat"], dtype=np.float32)
    # Build edge_index from G (directed)
    # convert to numpy edge list
    edges = np.array(list(G.edges()), dtype=np.int64).T  # shape [2, E]
    if edges.size == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.from_numpy(edges.copy()).long()

    data = Data(x=torch.from_numpy(feat), edge_index=edge_index)
    return data

# ---- Training loop that streams snapshots ----
def train_streaming(time_steps, temporal_edges, node_user_counts, spatial_edges, model, optimizer, device):
    """
    time_steps: sorted list of time indices to iterate over
    temporal_edges: dict t -> list of (u,v) movements from t->t+1; these edges will be added into snapshot G_t
    node_user_counts: dict t -> {node: count}
    spatial_edges: iterable of (u,v) static edges
    """
    model.train()
    # sliding buffer for seq_len inputs
    input_buffer = []
    edge_buffer = []

    pbar = tqdm(total=len(time_steps) * EPOCHS, desc="Epoch x Timestep", unit="step")

    for epoch in range(EPOCHS):
        # reset buffer for each epoch
        input_buffer.clear()
        edge_buffer.clear()

        for t in time_steps:
            # build snapshot graph G_t (nodes=0..NUM_NODES-1)
            G = nx.DiGraph()
            # add spatial edges once per snapshot
            G.add_edges_from(spatial_edges)
            # add temporal edges (movement edges originating at t)
            if t in temporal_edges:
                G.add_edges_from(temporal_edges[t])
            # add node features
            sin_t, cos_t = time_encoding(t)
            # only assign counts present; others stay zero
            counts = node_user_counts.get(t, {})
            # assign features
            for node in range(NUM_NODES):
                count = counts.get(node, 0)
                G.add_node(node, feat=[float(count), float(sin_t), float(cos_t)])

            # convert to pyg Data
            data = nx_to_pyg_data(G)
            data = data.to(device)

            # target: next timestep node counts (regression)
            # if t+1 not in node_user_counts, target is zeros
            next_counts_map = node_user_counts.get(t + 1, {})
            target = torch.zeros(NUM_NODES, dtype=torch.float32, device=device)
            if next_counts_map:
                # fill targets
                idxs = torch.tensor(list(next_counts_map.keys()), dtype=torch.long, device=device)
                vals = torch.tensor(list(next_counts_map.values()), dtype=torch.float32, device=device)
                target[idxs] = vals

            # add to buffer
            input_buffer.append(data.x)        # tensor [num_nodes, feat_dim]
            edge_buffer.append(data.edge_index)

            # keep only SEQ_LEN last items
            if len(input_buffer) > SEQ_LEN:
                input_buffer.pop(0)
                edge_buffer.pop(0)

            # only train once we have enough frames
            if len(input_buffer) == SEQ_LEN:
                optimizer.zero_grad()

                # For simplicity we use the latest snapshot edge_index for all recurrent steps.
                # (GConvGRU expects per-step X and a single edge_index)
                # stack inputs: take last input (current) as x; GConvGRU internal state is recurrent across calls
                # in this code we call recurrent once per time step (online style)
                x_cur = input_buffer[-1]  # [num_nodes, feat_dim]
                edge_index_cur = edge_buffer[-1]

                out = model(x_cur, edge_index_cur)  # [num_nodes]

                loss = F.mse_loss(out, target)
                loss.backward()
                optimizer.step()

            pbar.update(1)
        # end epoch
    pbar.close()

# ---- Example usage ----
def run_training(temporal_edges, node_user_counts, spatial_edges):
    time_steps = sorted(node_user_counts.keys())
    model = STGNN(NODE_FEAT_DIM, HIDDEN_DIM, OUT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    train_streaming(time_steps, temporal_edges, node_user_counts, spatial_edges, model, optimizer, DEVICE)
    return model
