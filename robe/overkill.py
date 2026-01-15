"""
Super-Simple Baseline (Time-Bigram + Occupancy)
===============================================
Updated for your schema:
  • Columns: uid, d (day 1..75), t (slot 0..47), x, y
  • No timestamps; time-of-day bin = t directly (B = 48 when 30‑min slots)

Goal
----
A tiny, robust baseline you can run on a mid-range laptop without ML frameworks.
It predicts 720 steps for masked users using:
  • A time-of-day bigram over relative moves (stay/move within radius R)
  • A per-time-bin occupancy prior over cells to prefer popular places
Streaming I/O only. No 40k×40k matrices. No errors.

Files per city (CSV or Parquet):
  city_A_trainmerged.(csv|parquet)  # days 1–60, all users
  city_A_testmerged.(csv|parquet)   # days 61–75, all users; masked users have x==MASK or y==MASK

Output:
  ./outputs/city_A_predictions.csv   with rows: uid,step,x,y (masked users only)

Run example (Windows):
  python simple_baseline.py ^
    --data_root C:\path\to\data ^
    --cities city_A city_B ^
    --grid_size 200 ^
    --radius 3 ^
    --time_freq_minutes 30 ^
    --mask_val 999
"""

import os
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Iterable, List

# ------------------------
# Helpers
# ------------------------

def encode_delta(dy: np.ndarray, dx: np.ndarray, R: int) -> np.ndarray:
    dyc = np.clip(dy, -R, R)
    dxc = np.clip(dx, -R, R)
    return (dyc + R) * (2*R + 1) + (dxc + R)


def decode_delta(idx: int, R: int) -> Tuple[int, int]:
    S = 2*R + 1
    dy = idx // S - R
    dx = idx % S - R
    return dy, dx


def to_cell_id(x: np.ndarray, y: np.ndarray, G: int) -> np.ndarray:
    return y * G + x


def from_cell_id(cell: int, G: int) -> Tuple[int, int]:
    y = cell // G
    x = cell % G
    return x, y


def clamp_xy(x: int, y: int, G: int) -> Tuple[int, int]:
    x = 0 if x < 0 else (G-1 if x >= G else x)
    y = 0 if y < 0 else (G-1 if y >= G else y)
    return x, y


# ------------------------
# Trainer (counts only)
# ------------------------

def fit_counts_for_city(
    path: str,
    grid_size: int,
    mask_val: int,
    time_freq_minutes: int,
    radius: int,
    chunksize: int = 200_000,
    user_col: str = "uid",
    day_col: str = "d",
    slot_col: str = "t",
    x_col: str = "x",
    y_col: str = "y",
) -> Tuple[np.ndarray, np.ndarray, Dict[int, Tuple[int, int]]]:
    """Return (delta_counts[B,K], occupancy[B,G*G], last_seen[user]=(last_cell, last_tbin)).
    B = number of time bins per day = 24*60 // time_freq_minutes (48 for 30 min slots)
    """
    B = 24*60 // time_freq_minutes
    K = (2*radius + 1) ** 2
    G = grid_size

    delta_counts = np.ones((B, K), dtype=np.float64)  # Laplace smoothing
    occupancy = np.zeros((B, G*G), dtype=np.int64)

    # Track last seen state per user across chunks: uid -> (last_cell, last_tbin)
    last_seen: Dict[int, Tuple[int, int]] = {}

    def process_df(df: pd.DataFrame):
        nonlocal delta_counts, occupancy, last_seen
        # enforce dtypes and sorting
        cols = [user_col, day_col, slot_col, x_col, y_col]
        df = df[cols].copy()
        df = df.sort_values([user_col, day_col, slot_col], kind='mergesort')
        # filter masked rows in train (should be none, but be defensive)
        m = (df[x_col].to_numpy() == mask_val) | (df[y_col].to_numpy() == mask_val)
        if m.any():
            df = df.loc[~m]
        if df.empty:
            return
        # compute bins and cells
        tbins = (df[slot_col].to_numpy(np.int16) % B)
        cells = to_cell_id(df[x_col].to_numpy(np.int32), df[y_col].to_numpy(np.int32), G)
        users = df[user_col].to_numpy(np.int64)

        # iterate per user group, preserving cross-chunk continuity via last_seen
        start = 0
        n = len(df)
        while start < n:
            uid = users[start]
            end = start + 1
            while end < n and users[end] == uid:
                end += 1
            u_cells = cells[start:end]
            u_bins = tbins[start:end]
            # occupancy for all rows
            np.add.at(occupancy, (u_bins, u_cells), 1)
            # transition from previous chunk tail to this chunk head
            prev = last_seen.get(int(uid))
            if prev is not None:
                prev_cell, prev_bin = prev
                dy = (u_cells[0] // G) - (prev_cell // G)
                dx = (u_cells[0] % G) - (prev_cell % G)
                didx = encode_delta(np.array([dy]), np.array([dx]), radius)[0]
                delta_counts[prev_bin, didx] += 1
            # transitions inside this chunk
            if end - start >= 2:
                dy = (u_cells[1:] // G) - (u_cells[:-1] // G)
                dx = (u_cells[1:] % G) - (u_cells[:-1] % G)
                didx_vec = encode_delta(dy, dx, radius)
                bins_prev = u_bins[:-1]
                np.add.at(delta_counts, (bins_prev, didx_vec), 1)
            # update last seen
            last_seen[int(uid)] = (int(u_cells[-1]), int(u_bins[-1]))
            start = end

    if path.endswith('.csv'):
        for chunk in pd.read_csv(path, chunksize=chunksize):
            process_df(chunk)
    else:
        df = pd.read_parquet(path, engine='pyarrow')
        process_df(df)

    return delta_counts, occupancy, last_seen


# ------------------------
# Inference
# ------------------------

def find_masked_users(path: str, mask_val: int, time_freq_minutes: int,
                       user_col: str, day_col: str, slot_col: str, x_col: str, y_col: str,
                       chunksize: int = 200_000) -> Dict[int, int]:
    """Return {uid: first_masked_tbin} for users with masked coords in test file.
    tbin here is simply the slot value (mod B).
    """
    B = 24*60 // time_freq_minutes
    masked: Dict[int, int] = {}

    def process_df(df: pd.DataFrame):
        nonlocal masked
        cols = [user_col, day_col, slot_col, x_col, y_col]
        df = df[cols].copy()
        df = df.sort_values([user_col, day_col, slot_col], kind='mergesort')
        tbins = (df[slot_col].to_numpy(np.int16) % B)
        m = (df[x_col].to_numpy() == mask_val) | (df[y_col].to_numpy() == mask_val)
        if not m.any():
            return
        users = df[user_col].to_numpy(np.int64)
        seen = set()
        for i in range(len(df)):
            if not m[i]:
                continue
            uid = int(users[i])
            if uid in seen or uid in masked:
                continue
            masked[uid] = int(tbins[i])
            seen.add(uid)

    if path.endswith('.csv'):
        for chunk in pd.read_csv(path, chunksize=chunksize):
            process_df(chunk)
    else:
        df = pd.read_parquet(path, engine='pyarrow')
        process_df(df)
    return masked


def rollout_users(masked_users: Dict[int, int], last_cell: Dict[int, Tuple[int, int]],
                  delta_counts: np.ndarray, occupancy: np.ndarray,
                  grid_size: int, radius: int, horizon: int) -> List[Tuple[int, np.ndarray]]:
    """Return list of (uid, predicted_cells[H])."""
    B = delta_counts.shape[0]
    G = grid_size
    # Normalize to probabilities
    delta_probs = delta_counts / delta_counts.sum(axis=1, keepdims=True)
    occ_probs = occupancy + 1  # avoid zero; we reweight candidates by local popularity

    out = []

    for uid, start_bin in masked_users.items():
        # Start state: last known cell from training (day<=60). If missing, pick most occupied cell at start_bin.
        if uid in last_cell:
            lastc, _ = last_cell[uid]
            cur_cell = int(lastc)
            cur_bin = int((start_bin - 1) % B)  # decision happens right before first masked slot
        else:
            cur_cell = int(occ_probs[start_bin].argmax())
            cur_bin = int((start_bin - 1) % B)
        preds = np.empty(horizon, dtype=np.int32)
        for h in range(horizon):
            p = delta_probs[cur_bin]
            next_bin = (cur_bin + 1) % B
            best_score = -1.0
            best_cell = cur_cell
            for didx in np.argsort(-p):
                dy, dx = decode_delta(int(didx), radius)
                x, y = from_cell_id(cur_cell, G)
                nx, ny = clamp_xy(x + dx, y + dy, G)
                ncell = ny * G + nx
                score = p[didx] * occ_probs[next_bin, ncell]
                if score > best_score:
                    best_score = score
                    best_cell = ncell
            cur_cell = int(best_cell)
            preds[h] = cur_cell
            cur_bin = next_bin
        out.append((uid, preds))
    return out


# ------------------------
# Writer
# ------------------------

def write_predictions(path_out: str, predictions: List[Tuple[int, np.ndarray]], grid_size: int):
    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    with open(path_out, 'w') as f:
        f.write('uid,step,x,y')
        for uid, cells in predictions:
            for h, c in enumerate(cells.tolist()):
                x, y = from_cell_id(c, grid_size)
                f.write(f"{uid},{h},{x},{y}")


# ------------------------
# CLI
# ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, required=True)
    ap.add_argument('--cities', nargs='+', required=True)
    ap.add_argument('--train_files', type=str, default='trainmerged')
    ap.add_argument('--test_files', type=str, default='testmerged')
    ap.add_argument('--grid_size', type=int, default=200)
    ap.add_argument('--time_freq_minutes', type=int, default=30)
    ap.add_argument('--radius', type=int, default=3)
    ap.add_argument('--horizon', type=int, default=720)
    ap.add_argument('--mask_val', type=int, default=999)
    ap.add_argument('--chunksize', type=int, default=200_000)
    ap.add_argument('--user_col', type=str, default='uid')
    ap.add_argument('--day_col', type=str, default='d')
    ap.add_argument('--slot_col', type=str, default='t')
    ap.add_argument('--x_col', type=str, default='x')
    ap.add_argument('--y_col', type=str, default='y')
    ap.add_argument('--out_dir', type=str, default='outputs')
    args = ap.parse_args()

    G = args.grid_size

    for city in args.cities:
        print(f"[+] City {city}: fitting counts from train...")
        train_path_csv = os.path.join(args.data_root, f"{city}_{args.train_files}.csv")
        train_path_parq = os.path.join(args.data_root, f"{city}_{args.train_files}.parquet")
        train_path = train_path_csv if os.path.exists(train_path_csv) else train_path_parq
        if not os.path.exists(train_path):
            raise FileNotFoundError(train_path_csv + ' or ' + train_path_parq)

        delta_counts, occupancy, last_seen = fit_counts_for_city(
            path=train_path,
            grid_size=G,
            mask_val=args.mask_val,
            time_freq_minutes=args.time_freq_minutes,
            radius=args.radius,
            chunksize=args.chunksize,
            user_col=args.user_col,
            day_col=args.day_col,
            slot_col=args.slot_col,
            x_col=args.x_col,
            y_col=args.y_col,
        )

        print(f"[+] City {city}: finding masked users in test...")
        test_path_csv = os.path.join(args.data_root, f"{city}_{args.test_files}.csv")
        test_path_parq = os.path.join(args.data_root, f"{city}_{args.test_files}.parquet")
        test_path = test_path_csv if os.path.exists(test_path_csv) else test_path_parq
        if not os.path.exists(test_path):
            raise FileNotFoundError(test_path_csv + ' or ' + test_path_parq)

        masked = find_masked_users(
            path=test_path,
            mask_val=args.mask_val,
            time_freq_minutes=args.time_freq_minutes,
            user_col=args.user_col,
            day_col=args.day_col,
            slot_col=args.slot_col,
            x_col=args.x_col,
            y_col=args.y_col,
            chunksize=args.chunksize,
        )
        if not masked:
            print(f"[!] No masked users detected in {city}_{args.test_files}.")

        print(f"[+] City {city}: rolling out predictions for {len(masked)} masked users...")
        preds = rollout_users(masked, last_seen, delta_counts, occupancy,
                              grid_size=G, radius=args.radius, horizon=args.horizon)

        out_path = os.path.join(args.out_dir, f"{city}_predictions.csv")
        write_predictions(out_path, preds, grid_size=G)
        print(f"[✓] Saved {out_path}")


if __name__ == '__main__':
    main()
