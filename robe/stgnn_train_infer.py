import os
import argparse
import math
import json
import random
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv

# Only for testing with reduced dataset
DEFAULT_USE_TEST_FOLDER = True

# You can toggle this if PyG is painful to install.
DEFAULT_USE_PYG = True

try:
    import torch_geometric
    from torch_geometric.nn import GCNConv
    PYG_AVAILABLE = True
except Exception:
    PYG_AVAILABLE = False

def seed_all(seed=42):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def read_city(data_dir, city):
    train_path = os.path.join(data_dir, f"city_{city}_trainmerged.csv")
    test_path  = os.path.join(data_dir, f"city_{city}_testmerged.csv")
    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)
    return df_train, df_test

def normalize_xy(df):
    # grid is 0..199 per challenge spec
    df = df.copy()
    df['x_norm'] = df['x'] / 199.0
    df['y_norm'] = df['y'] / 199.0
    # handle masked rows (999) by NaN
    df.loc[df['x'] >= 900, ['x_norm','y_norm']] = np.nan
    return df

def compute_home_xy(train_df):
    # home = modal cell over days 1-60
    homes = {}
    for uid, g in train_df.groupby('uid'):
        mode_x = g['x'].mode()
        mode_y = g['y'].mode()
        if len(mode_x)==0 or len(mode_y)==0:
            homes[uid] = (g['x'].median(), g['y'].median())
        else:
            homes[uid] = (int(mode_x.iloc[0]), int(mode_y.iloc[0]))
    return homes

def build_user_graph(homes, k=10):
    # Build k-NN over home coordinates
    uids = np.array(list(homes.keys()))
    coords = np.array([homes[u] for u in uids], dtype=float)
    if len(uids) == 0:
        return uids, np.zeros((0,2)), np.zeros((2,0), dtype=np.int64)
    nn = NearestNeighbors(n_neighbors=min(k+1, len(uids)), metric='euclidean')
    nn.fit(coords)
    dists, inds = nn.kneighbors(coords, return_distance=True)
    # Build undirected edges excluding self
    edges = []
    for i, nbrs in enumerate(inds):
        for j in nbrs[1:]:
            edges.append((i, j))
            edges.append((j, i))
    edges = np.array(edges, dtype=np.int64).T if len(edges)>0 else np.zeros((2,0), dtype=np.int64)
    return uids, coords, edges

def make_sequences(df_train, history=6):
    # returns dict uid -> {(d,t) -> feature}, and list of (uid, d, t, target_x, target_y)
    # feature at time (d,t) is [x_norm, y_norm, sin_time, cos_time]
    seq_feats = {}
    targets = []
    # Build lookup
    df = df_train.copy()
    df = df.sort_values(['uid','d','t'])
    for uid, g in tqdm(df.groupby('uid'), desc="features"):
        idx = {}
        xs = g['x_norm'].values
        ys = g['y_norm'].values
        ds = g['d'].values
        ts = g['t'].values
        for i in range(len(g)):
            d, t = int(ds[i]), int(ts[i])
            # time enc
            time_index = (d-1)*48 + t
            sin_time = math.sin(2*math.pi * (time_index % 48) / 48.0)
            cos_time = math.cos(2*math.pi * (time_index % 48) / 48.0)
            feat = np.array([xs[i] if not np.isnan(xs[i]) else 0.5,
                             ys[i] if not np.isnan(ys[i]) else 0.5,
                             sin_time, cos_time], dtype=np.float32)
            idx[(d,t)] = feat
            # next step target if exists
        seq_feats[uid] = idx
    return seq_feats

def get_tensor_batch(uids, seq_feats, d, t, device):
    # build node feature tensor X: [num_nodes, feat_dim]
    feats = []
    for uid in uids:
        f = seq_feats.get(uid, {})
        feat = f.get((d,t), np.array([0.5,0.5,0.0,1.0], dtype=np.float32))
        feats.append(feat)
    X = torch.tensor(np.stack(feats, axis=0), dtype=torch.float32, device=device)
    return X

class SimpleTGNN(nn.Module):
    # Pure-PyTorch fallback: GraphConv via adjacency matmul + GRU
    def __init__(self, in_dim=4, hidden=64, out_dim=2):
        super().__init__()
        self.hidden = hidden
        self.lin_in = nn.Linear(in_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.lin_out = nn.Linear(hidden, out_dim)

    def forward(self, X_seq, A_hat):
        # X_seq: [B, T, F], A_hat: [B, B] normalized adjacency applied per time step
        H = self.lin_in(X_seq)  # [B,T,H]
        # message passing: aggregate neighbors at each time slice
        # we do a quick-and-dirty A_hat @ x_t
        B, T, Hdim = H.shape
        agg = []
        for t in range(T):
            xt = H[:, t, :]  # [B,H]
            xt_agg = torch.matmul(A_hat, xt)  # [B,H]
            agg.append(xt_agg.unsqueeze(1))
        Hagg = torch.cat(agg, dim=1)  # [B,T,H]
        out, _ = self.gru(Hagg)  # [B,T,H]
        y = self.lin_out(out[:, -1, :])  # last step -> [B,2]
        return y

class PyGTGNN(nn.Module):
    # PyG version: GCNConv + GRU over time
    def __init__(self, num_nodes, in_dim=4, hidden=64, out_dim=2, edge_index=None):
        super().__init__()
        self.gcn1 = GCNConv(in_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.lin = nn.Linear(hidden, out_dim)
        self.register_buffer('edge_index', edge_index if edge_index is not None else torch.zeros((2,0), dtype=torch.long))

    def forward(self, X_seq):
        # X_seq: [B, T, F]; but GCNConv expects [N,F]; we'll run per time slice
        B, T, Fdim = X_seq.shape
        device = X_seq.device
        Hs = []
        for t in range(T):
            xt = X_seq[:, t, :]
            ht = self.gcn1(xt, self.edge_index)
            ht = F.relu(ht)
            Hs.append(ht.unsqueeze(1))
        H = torch.cat(Hs, dim=1)
        out, _ = self.gru(H)
        y = self.lin(out[:, -1, :])
        return y

def build_adj_from_edges(num_nodes, edges, device):
    if edges.shape[1] == 0:
        A = torch.eye(num_nodes, device=device)
    else:
        idx_i = torch.tensor(edges[0], dtype=torch.long, device=device)
        idx_j = torch.tensor(edges[1], dtype=torch.long, device=device)
        A = torch.zeros((num_nodes, num_nodes), device=device)
        A[idx_i, idx_j] = 1.0
        A[idx_j, idx_i] = 1.0
        A = A + torch.eye(num_nodes, device=device)  # self
    # normalize A_hat = D^{-1} A
    deg = A.sum(dim=1, keepdim=True) + 1e-6
    A_hat = A / deg
    return A_hat

def train_city(args, city):
    print(f"\n=== City {city} ===")
    df_train, df_test = read_city(args.data_dir, city)
    # Filter days as a sanity check
    df_train = df_train[df_train['d'] <= 60].copy()
    df_test = df_test[(df_test['d'] >= 61) & (df_test['d'] <= 75)].copy()

    # Normalize and compute homes
    df_train = normalize_xy(df_train)
    df_test  = normalize_xy(df_test)

    homes = compute_home_xy(df_train)
    uids_all = np.array(sorted(df_train['uid'].unique()))
    # Build graph
    uids_graph, coords, edges = build_user_graph(homes, k=args.knn)
    uid_to_idx = {u:i for i,u in enumerate(uids_graph)}

    # Prepare sequences for training (days 1..60)
    seq_feats = make_sequences(df_train, history=args.history)

    # Training data: for each time (d,t) with d in [1..60-history), build a window
    # To keep things lightweight, we sample a subset of timesteps.
    all_dt = sorted(set((int(d),int(t)) for d,t in zip(df_train['d'], df_train['t'])))
    # we want windows that have a valid (d,t+1) inside train
    train_windows = []
    for d, t in all_dt:
        if d == 60 and t == 47:  # no next
            continue
        # Target exists in train set only if d_next<=60
        d_next, t_next = (d, t+1) if t < 47 else (d+1, 0)
        if d_next <= 60:
            train_windows.append((d, t, d_next, t_next))

    # device
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    # Model
    num_nodes = len(uids_graph)
    if num_nodes == 0:
        print("No users found; skipping city.")
        return None

    if args.use_pyg and PYG_AVAILABLE:
        edge_index = torch.tensor(edges, dtype=torch.long, device=device) if edges.shape[1] > 0 else torch.zeros((2,0), dtype=torch.long, device=device)
        model = PyGTGNN(num_nodes, in_dim=4, hidden=args.hidden, out_dim=2, edge_index=edge_index).to(device)
        use_adj = False
        A_hat = None
    else:
        model = SimpleTGNN(in_dim=4, hidden=args.hidden, out_dim=2).to(device)
        use_adj = True
        A_hat = build_adj_from_edges(num_nodes, edges, device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.SmoothL1Loss()

    # Training loop (tiny, for demonstration)
    model.train()
    steps = min(len(train_windows), args.max_steps) if args.max_steps > 0 else len(train_windows)
    history = args.history
    for epoch in range(args.epochs):
        print(f"Training, epoch {epoch+1}/{args.epochs}...")
        random.shuffle(train_windows)
        epoch_loss = 0.0
        for i in range(steps):
            d, t, d_next, t_next = train_windows[i]
            # Build X sequence [N, history, F]
            X_seq = []
            for h in range(history):
                dd = d
                tt = t - (history-1-h)
                while tt < 0:
                    dd -= 1
                    tt += 48
                X_seq.append(get_tensor_batch(uids_graph, seq_feats, dd, tt, device))
            X_seq = torch.stack(X_seq, dim=1)  # [N, H, F]

            # Targets from next step
            y_next = get_tensor_batch(uids_graph, seq_feats, d_next, t_next, device)[:, :2]

            opt.zero_grad()
            if use_adj:
                y_hat = model(X_seq, A_hat)
            else:
                y_hat = model(X_seq)
            loss = loss_fn(y_hat, y_next)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs} done - Loss {epoch_loss/max(1,steps):.5f}")

    # Inference: roll over days 61..75, t=0..47
    # Build combined seq_feats including test, but with masked users filled on-the-fly by predictions
    seq_feats_test = make_sequences(pd.concat([df_train, df_test], ignore_index=True), history=args.history)

    # Determine masked users in test: last 3000 uids present in test with x=999 on day 61
    # Robust rule: user is masked if any test row has x>=900
    masked_uids = set(df_test.loc[df_test['x']>=900, 'uid'].unique().tolist())

    out_rows = []
    model.eval()
    with torch.no_grad():
        for d in range(61, 76):
            for t in range(48):
                # assemble input sequence for current (d,t) from H previous steps
                X_seq = []
                for h in range(args.history):
                    dd = d
                    tt = t - (args.history-1-h)
                    while tt < 0:
                        dd -= 1
                        tt += 48
                    X_seq.append(get_tensor_batch(uids_graph, seq_feats_test, dd, tt, device))
                X_seq = torch.stack(X_seq, dim=1)

                if use_adj:
                    y_hat = model(X_seq, A_hat)
                else:
                    y_hat = model(X_seq)

                # y_hat is normalized [0,1], map back to grid and write for masked only
                y_hat = y_hat.clamp(0,1).cpu().numpy()
                x_pred = np.rint(y_hat[:,0] * 199).astype(int)
                y_pred = np.rint(y_hat[:,1] * 199).astype(int)

                # Update seq_feats_test for masked users so their future inputs use these predictions
                for idx, uid in enumerate(uids_graph):
                    if uid in masked_uids:
                        sin_time = math.sin(2*math.pi * ((d-1)*48 + t) / 48.0)
                        cos_time = math.cos(2*math.pi * ((d-1)*48 + t) / 48.0)
                        seq_feats_test.setdefault(uid, {})[(d,t)] = np.array([x_pred[idx]/199.0, y_pred[idx]/199.0, sin_time, cos_time], dtype=np.float32)
                        out_rows.append((int(uid), d, t, int(x_pred[idx]), int(y_pred[idx])))

    out_df = pd.DataFrame(out_rows, columns=['uid','d','t','x_pred','y_pred'])
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"city_{city}_predictions.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Wrote predictions to {out_path}")
    return out_path

def main():
    default_data_dir = './dataset/preproc/test' if DEFAULT_USE_TEST_FOLDER else './dataset/preproc'

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=default_data_dir)
    parser.add_argument('--out_dir', type=str, default='./predictions')
    parser.add_argument('--cities', nargs='+', default=['A','B','C','D'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--history', type=int, default=6)
    parser.add_argument('--knn', type=int, default=10)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_steps', type=int, default=2000, help='limit training steps for speed; -1 for all')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--use_pyg', type=int, default=-1, help='1 to force PyG, 0 to force fallback, -1 to auto')
    args = parser.parse_args()

    seed_all(42)

    if args.use_pyg == 1:
        use_pyg = True
    elif args.use_pyg == 0:
        use_pyg = False
    else:
        use_pyg = DEFAULT_USE_PYG and PYG_AVAILABLE
    args.use_pyg = use_pyg

    os.makedirs(args.out_dir, exist_ok=True)
    for city in args.cities:
        try:
            train_city(args, city)
        except FileNotFoundError as e:
            print(f"Skipping city {city}: {e}")

if __name__ == "__main__":
    main()
