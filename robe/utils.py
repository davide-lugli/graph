import numpy as np

# Cell mapping
def cell_to_node_id(x, y):
    return x * 200 + y

def node_id_to_cell(node_id):
    return divmod(node_id, 200)

# Time encoding for periodic features
def time_encoding(t):
    angle = 2 * np.pi * (t % 48) / 48
    return np.sin(angle), np.cos(angle)
