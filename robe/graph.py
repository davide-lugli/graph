import os
import pickle
import pandas as pd
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
from utils import cell_to_node_id, node_id_to_cell, time_encoding

train_filename = "city_A"

### 1. Load train data
### 2. Define Cell to Node ID Mapping
'''
- Loads your train data from CSV if no cached processed file exists.
- Keeps only relevant columns: user ID (uid), day (d), time interval (t), and coordinates (x, y).
- Creates a unified time index time_idx combining day and time intervals.
- Maps each cell (x, y) to a unique node_id.
- Saves processed data as a parquet file for faster loading next time
'''
processed_path = f"dataset/step_1_2_proc/{train_filename}_train_processed.parquet"
if os.path.exists(processed_path):
    print("Skipping Steps #1 and #2 by using preprocessed data...")
    train_df = pd.read_parquet(processed_path)
else:
    print("Step #1 : Loading and processing train data...")
    train_df = pd.read_csv(f"dataset/preproc/{train_filename}_trainmerged.csv")
    train_df = train_df[["uid", "d", "t", "x", "y"]]
    # train_df = train_df[(train_df['x'] < 9999) & (train_df['y'] < 9999)]
    train_df["time_idx"] = (train_df["d"] - 1) * 48 + train_df["t"]

    print("Step #2 : Defining Cell to Node ID Mapping...")
    train_df["node_id"] = train_df.apply(lambda row: cell_to_node_id(row["x"], row["y"]), axis=1)

    print("Saving processed data for future runs...")
    train_df.to_parquet(processed_path, index=False)

### 3. Build Temporal Transitions (User Movement)
'''
- For each user, sorts their records in time order.
- Builds edges between consecutive nodes visited by the user at each time step — these are your temporal edges.
- Counts how many users are at each node at each time step (node_user_counts).
'''
print("Step #3 : Building temporal edges and node user counts for all time steps...")
df_sorted = train_df.sort_values(["uid", "time_idx"])
uids = df_sorted["uid"].to_numpy()
nodes = df_sorted["node_id"].to_numpy()
times = df_sorted["time_idx"].to_numpy()

same_user = (uids[1:] == uids[:-1])
u_nodes = nodes[:-1][same_user]
v_nodes = nodes[1:][same_user]
t_times = times[:-1][same_user]

temporal_edges = defaultdict(list)
for u, v, t in zip(u_nodes, v_nodes, t_times):
    temporal_edges[t].append((u, v))

node_user_counts_df = df_sorted.groupby(["time_idx", "node_id"]).size().reset_index(name="count")
node_user_counts = defaultdict(dict)
for _, row in node_user_counts_df.iterrows():
    node_user_counts[row["time_idx"]][row["node_id"]] = row["count"]


### 4. Build Spatial Edges (Once for all time)
'''
- Builds edges between each cell and its 8 neighbors (up, down, left, right, and diagonals).
- This graph is static, representing spatial adjacency of cells.
'''
print("Step #4 : Building spatial edges...")
spatial_edges = set()
for x in range(200):
    for y in range(200):
        center = cell_to_node_id(x, y)
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
            nx_, ny_ = x + dx, y + dy
            if 0 <= nx_ < 200 and 0 <= ny_ < 200:
                neighbor = cell_to_node_id(nx_, ny_)
                spatial_edges.add((center, neighbor))

### 5. Build Graph Snapshots for ST-GNN
'''
- Creates a directed graph with spatial edges plus temporal edges for all timesteps.
- Adds node features: user count at that time + cyclical time encoding.
'''
print("Step #5 : Building graphs and training model...")
time_steps = sorted(node_user_counts.keys())

tqdm.set_postfix({"loss": loss.item()})
for t in tqdm(time_steps, desc="Training Progress"):
    G = nx.DiGraph()
    G.add_edges_from(spatial_edges)
    if t in temporal_edges:
        G.add_edges_from(temporal_edges[t])

    sin_t, cos_t = time_encoding(t)
    for node in G.nodes():
        count = node_user_counts[t].get(node, 0)
        G.nodes[node]["feat"] = [count, sin_t, cos_t]

    # 🔹 Directly pass graph to your training step
    train_on_graph_snapshot(G, t)  # <- you implement this

