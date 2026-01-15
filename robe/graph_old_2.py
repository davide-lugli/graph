import os
import pickle
import pandas as pd
import networkx as nx
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
temporal_edges_path = f"dataset/step_3_proc/{train_filename}_temporal_edges.pkl"
node_user_counts_path = f"dataset/step_3_proc/{train_filename}_node_user_counts.pkl"

if os.path.exists(temporal_edges_path) and os.path.exists(node_user_counts_path):
    print("Skipping Step #3 by using precomputed temporal edges and node user counts...")
    with open(temporal_edges_path, "rb") as f:
        temporal_edges = pickle.load(f)
    with open(node_user_counts_path, "rb") as f:
        node_user_counts = pickle.load(f)
else:
    print("Step #3 : Building temporal edges and node user counts for all time steps...")
    df_sorted = train_df.sort_values(["uid", "time_idx"])
    uids = df_sorted["uid"].to_numpy()
    nodes = df_sorted["node_id"].to_numpy()
    times = df_sorted["time_idx"].to_numpy()

    # Mask: identify rows where the next row is same user (to build edges)
    same_user = (uids[1:] == uids[:-1])

    # Temporal edges: (u,v) at time t
    u_nodes = nodes[:-1][same_user]
    v_nodes = nodes[1:][same_user]
    t_times = times[:-1][same_user]

    temporal_edges = defaultdict(list)
    for u, v, t in zip(u_nodes, v_nodes, t_times):
        temporal_edges[t].append((u, v))

    # Use pandas groupby for fast counts
    node_user_counts_df = df_sorted.groupby(["time_idx", "node_id"]).size().reset_index(name="count")

    # Convert to nested dict: node_user_counts[time][node] = count
    node_user_counts = defaultdict(dict)
    for _, row in node_user_counts_df.iterrows():
        node_user_counts[row["time_idx"]][row["node_id"]] = row["count"]

    print("Saving temporal edges and node user counts for future use...")
    with open(temporal_edges_path, "wb") as f:
        pickle.dump(dict(temporal_edges), f)

    with open(node_user_counts_path, "wb") as f:
        pickle.dump(dict(node_user_counts), f)

### 4. Build Spatial Edges (Once for all time)
'''
- Builds edges between each cell and its 8 neighbors (up, down, left, right, and diagonals).
- This graph is static, representing spatial adjacency of cells.
'''
spatial_edges_path = f"dataset/step_4_proc/{train_filename}_spatial_edges.pkl"

if os.path.exists(spatial_edges_path):
    print("Skipping Step #4 by loading cached spatial edges...")
    with open(spatial_edges_path, "rb") as f:
        spatial_edges = pickle.load(f)
else:
    print("Step #4 : Building spatial edges...")
    spatial_edges = set()
    for x in range(200):
        for y in range(200):
            center = cell_to_node_id(x, y)
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:  # 8-connected
                nx_, ny_ = x + dx, y + dy
                if 0 <= nx_ < 200 and 0 <= ny_ < 200:
                    neighbor = cell_to_node_id(nx_, ny_)
                    spatial_edges.add((center, neighbor))

    print("Saving spatial edges for future use...")
    os.makedirs(os.path.dirname(spatial_edges_path), exist_ok=True)
    with open(spatial_edges_path, "wb") as f:
        pickle.dump(spatial_edges, f)

### 5. Build Graph Snapshots for ST-GNN (Example)
'''
- Creates a directed graph with spatial edges plus temporal edges for all timesteps.
- Adds node features: user count at that time + cyclical time encoding.
'''
graphs_dir = f"dataset/graphs/{train_filename}/"
os.makedirs(graphs_dir, exist_ok=True)

print("Step #5 : Building and saving graph snapshots for all time steps...")
time_steps = sorted(node_user_counts.keys())

for t in time_steps:
    G = nx.DiGraph()
    # Add spatial edges
    G.add_edges_from(spatial_edges)
    # Add temporal edges for this time step (user movements)
    if t in temporal_edges:
        G.add_edges_from(temporal_edges[t])

    # Add node features: user counts + time encoding
    sin_t, cos_t = time_encoding(t)
    for node in G.nodes():
        count = node_user_counts[t].get(node, 0)
        G.nodes[node]["feat"] = [count, sin_t, cos_t]

    # Save graph (can be pickle or any preferred format)
    nx.write_gpickle(G, os.path.join(graphs_dir, f"graph_{t}.gpickle"))

print(f"Saved {len(time_steps)} graph snapshots to {graphs_dir}")
