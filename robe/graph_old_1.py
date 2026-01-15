import os
import pandas as pd
import numpy as np
import networkx as netx
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
print("Step #1 : Loading train data...")
processed_path = f"dataset/step1_proc/{train_filename}_train_processed.parquet"
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
print("Step #3 : Building Temporal Transitions...")
# Store edges per time step
temporal_edges = defaultdict(list)
# Store user counts per node per time step
node_user_counts = defaultdict(lambda: defaultdict(int))  # node_user_counts[time][node] = count

for uid, group in train_df.groupby("uid"):
    group_sorted = group.sort_values("time_idx")
    nodes = group_sorted["node_id"].tolist()
    times = group_sorted["time_idx"].tolist()

    for i in range(len(nodes) - 1):
        u = nodes[i]
        v = nodes[i + 1]
        t = times[i]
        temporal_edges[t].append((u, v))

    for t, node in zip(times, nodes):
        node_user_counts[t][node] += 1

### 4. Build Spatial Edges (Once for all time)
'''
- Builds edges between each cell and its 8 neighbors (up, down, left, right, and diagonals).
- This graph is static, representing spatial adjacency of cells.
'''
print("Step #4 : Building Spatial Edges...")
spatial_edges = set()
for x in range(200):
    for y in range(200):
        center = cell_to_node_id(x, y)
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:  # 8-connected
            nx, ny = x + dx, y + dy
            if 0 <= nx < 200 and 0 <= ny < 200:
                neighbor = cell_to_node_id(nx, ny)
                spatial_edges.add((center, neighbor))

### 5. Build Graph Snapshots for ST-GNN (Example)
print("Step #5 : Building Graph Snapshots...")
time_idx = 1000  # pick any valid one from your data
G = netx.DiGraph()
# Add spatial edges
G.add_edges_from(spatial_edges)
# Add temporal movement edges
G.add_edges_from(temporal_edges[time_idx])

# Add node features: user count and time-of-day
for node in G.nodes:
    count = node_user_counts[time_idx].get(node, 0)
    sin_t, cos_t = time_encoding(time_idx)
    G.nodes[node]["feat"] = [count, sin_t, cos_t]

# Inspect a few node features
for node in list(G.nodes)[:5]:
    print(f"Node {node}: {G.nodes[node]}")
