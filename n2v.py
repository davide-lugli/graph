import os
import gzip
import pickle
import hashlib
import geobleu
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import networkx as nx
from node2vec import Node2Vec
from tqdm import tqdm

CITIES = ['A', 'B', 'C', 'D']
SKIP_GEOBLEU = True
GRID = 200
CELL_SIZE_M = 500  # 500m per cella

### HELPERS

def cell_id(x, y):
    # x,y in [1..200]
    return (x - 1) * GRID + (y - 1)

def decode_cell(c):
    x = c // GRID + 1
    y = c % GRID + 1
    return x, y

def weekday_of(d):
    return (d - 1) % 7

def context_id(d, t):
    return weekday_of(d) * 48 + t  # 0..335

def file_fingerprint(path, block_size=1 << 20):
    """
    Fingerprint veloce e abbastanza robusta:
    - size
    - mtime
    - hash primi+ultimi blocchi
    """
    st = os.stat(path)
    size = st.st_size
    mtime = int(st.st_mtime)

    h = hashlib.sha256()
    h.update(str(size).encode())
    h.update(str(mtime).encode())

    with open(path, "rb") as f:
        head = f.read(block_size)
        h.update(head)
        if size > block_size:
            f.seek(max(0, size - block_size))
            tail = f.read(block_size)
            h.update(tail)

    return h.hexdigest()

def cache_path_for(city, train_path, cache_dir="cache", tag="n2v_proto"):
    os.makedirs(cache_dir, exist_ok=True)
    fp = file_fingerprint(train_path)
    return os.path.join(cache_dir, f"{tag}_city_{city}_{fp}.pkl.gz")

def save_cache(path, payload):
    with gzip.open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL) # type: ignore

def load_cache(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f) # type: ignore

### PREDICTION

def build_cell_graph(train_df):
    # expects columns: uid,d,t,x,y
    train_df = train_df.sort_values(["uid", "d", "t"])
    x = train_df["x"].to_numpy()
    y = train_df["y"].to_numpy()
    uid = train_df["uid"].to_numpy()

    cells = np.array([cell_id(int(a), int(b)) for a, b in zip(x, y)], dtype=np.int32)

    # consecutive rows within same uid are consecutive timesteps (thanks to your preprocessing)
    same_user = uid[1:] == uid[:-1]
    src = cells[:-1][same_user]
    dst = cells[1:][same_user]

    # count weighted edges
    edge_counts = Counter(zip(src.tolist(), dst.tolist()))

    G = nx.DiGraph()
    for (u, v), w in edge_counts.items():
        if u == v:
            continue
        G.add_edge(u, v, weight=w)
    return G

def train_node2vec(G, dim=64, walk_length=20, num_walks=10, window=10, workers=4):
    if os.name == "nt":
        workers = 1

    n2v = Node2Vec(
        G,
        dimensions=dim,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=workers,
        weight_key="weight"
    )
    model = n2v.fit(window=window, min_count=1, batch_words=4096)
    emb = {int(n): model.wv[str(n)].astype(np.float32) for n in G.nodes()}
    return emb

def build_prototypes(train_df, emb):
    # proto[(uid, context)] = mean embedding
    buckets = defaultdict(list)

    for r in train_df.itertuples(index=False):
        c = cell_id(int(r.x), int(r.y))
        if c not in emb:
            continue
        ctx = context_id(int(r.d), int(r.t))
        buckets[(int(r.uid), ctx)].append(emb[c])

    proto = {}
    for k, vecs in buckets.items():
        proto[k] = np.mean(vecs, axis=0)
    return proto

def top_cells_by_user(train_df, topN=200):
    out = {}
    for uid, g in train_df.groupby("uid"):
        cells = [cell_id(int(x), int(y)) for x, y in zip(g["x"], g["y"])]
        out[int(uid)] = [c for c, _ in Counter(cells).most_common(topN)]
    return out

def top_cells_by_context(train_df, topM=200):
    out = {}
    # compute per context popularity
    ctx_cells = defaultdict(list)
    for r in train_df.itertuples(index=False):
        ctx = context_id(int(r.d), int(r.t))
        ctx_cells[ctx].append(cell_id(int(r.x), int(r.y)))
    for ctx, cells in ctx_cells.items():
        out[ctx] = [c for c, _ in Counter(cells).most_common(topM)]
    return out

def prepare_embedding_index(emb: dict):
    """
    Prepara strutture per cosine_best fast:
    - node_list: array dei node ids (celle)
    - id2idx: mapping cell_id -> row index
    - E: embedding matrix (N, dim)
    - E_norm: norma per ogni embedding (N,)
    """
    node_list = np.array(list(emb.keys()), dtype=np.int32)
    id2idx = {int(n): i for i, n in enumerate(node_list)}
    E = np.stack([emb[int(n)] for n in node_list]).astype(np.float32)
    E_norm = np.linalg.norm(E, axis=1).astype(np.float32) + 1e-9
    return node_list, id2idx, E, E_norm

def cosine_best(proto_vec, candidates, id2idx, E, E_norm, node_list):
    """
    Ritorna il candidate (cell_id) con massima cosine similarity con proto_vec.
    Versione veloce: fa tutto in NumPy (no loop Python sui candidati).
    
    - proto_vec: np.array (dim,)
    - candidates: iterable di cell_id (int)
    - id2idx: dict cell_id -> idx in E
    - E: np.ndarray (N, dim) embeddings
    - E_norm: np.ndarray (N,) norme embeddings
    - node_list: np.ndarray (N,) cell_id per idx
    """
    if proto_vec is None:
        return None

    # filtra candidati presenti in emb
    cand_idx = []
    for c in candidates:
        i = id2idx.get(int(c))
        if i is not None:
            cand_idx.append(i)

    if not cand_idx:
        return None

    cand_idx = np.array(cand_idx, dtype=np.int32)

    pv = np.asarray(proto_vec, dtype=np.float32)
    pv_norm = float(np.linalg.norm(pv) + 1e-9)

    # cosine: (E[c] @ pv) / (||E[c]|| * ||pv||)
    scores = (E[cand_idx] @ pv) / (E_norm[cand_idx] * pv_norm)

    best_local = int(np.argmax(scores))
    best_idx = int(cand_idx[best_local])
    return int(node_list[best_idx])

def cosine_best_old(proto_vec, candidates, emb):
    # return candidate with max cosine similarity
    pv = proto_vec
    pv_norm = np.linalg.norm(pv) + 1e-9
    best_c, best_s = None, -1e9
    for c in candidates:
        v = emb.get(c)
        if v is None:
            continue
        s = float(np.dot(pv, v) / (pv_norm * (np.linalg.norm(v) + 1e-9)))
        if s > best_s:
            best_s, best_c = s, c
    return best_c

def run_city(city, city_train_path, city_test_path, out_path,
             fill_masked_only=False,
             use_cache=True,
             cache_dir="cache",
             n2v_dim=64,
             n2v_walk_length=20,
             n2v_num_walks=10,
             n2v_window=10,
             n2v_workers=4,
             topN_user=200,
             topM_ctx=200):
    print(f"Running city pipeline for CITY={city}")
    train = pd.read_csv(city_train_path)
    test  = pd.read_csv(city_test_path)

    # keep a copy of original GT for evaluation masking
    test_gt = test.copy()

    # masked users = quelli che nel test hanno almeno una riga con 999
    masked_uids = set(test_gt.loc[(test_gt["x"] == 999) | (test_gt["y"] == 999), "uid"].unique())

    # --------------------
    # TRAIN ON ALL USERS AND CACHE
    # --------------------
    cache_file = cache_path_for(city, city_train_path, cache_dir=cache_dir)
    cache_key = {
        "CITY": city,
        "GRID": GRID,
        "CELL_SIZE_M": CELL_SIZE_M,
        "n2v_dim": n2v_dim,
        "n2v_walk_length": n2v_walk_length,
        "n2v_num_walks": n2v_num_walks,
        "n2v_window": n2v_window,
        "n2v_workers": n2v_workers,
        "topN_user": topN_user,
        "topM_ctx": topM_ctx,
    }

    emb = None
    proto = None
    user_top = None
    ctx_top = None
    user_mode = None
    if use_cache and os.path.exists(cache_file):
        payload = load_cache(cache_file)
        if payload.get("cache_key") == cache_key:
            emb = payload["emb"]
            proto = payload["proto"]
            user_top = payload["user_top"]
            ctx_top = payload["ctx_top"]
            user_mode = payload["user_mode"]
            print(f"Loaded cache: {cache_file}")
        else:
            print("Cache found but params mismatch, rebuilding...")
            payload = None
    else:
        payload = None

    if payload is None:
        train_used = train  # all users

        G = build_cell_graph(train_used)
        emb = train_node2vec(
            G,
            dim=n2v_dim,
            walk_length=n2v_walk_length,
            num_walks=n2v_num_walks,
            window=n2v_window,
            workers=n2v_workers
        )
        print("Graph and embeddings built.")

        proto = build_prototypes(train_used, emb)
        user_top = top_cells_by_user(train_used, topN=topN_user)
        ctx_top  = top_cells_by_context(train_used, topM=topM_ctx)

        # fallback per-user mode
        user_mode = {}
        for uid, g in train_used.groupby("uid"):
            cells = [cell_id(int(x), int(y)) for x, y in zip(g["x"], g["y"])]
            user_mode[int(uid)] = Counter(cells).most_common(1)[0][0] # type: ignore

        print("Prototypes and candidate pools built.")

        if use_cache:
            save_cache(cache_file, {
                "cache_key": cache_key,
                "emb": emb,
                "proto": proto,
                "user_top": user_top,
                "ctx_top": ctx_top,
                "user_mode": user_mode
            })
            print(f"Saved cache: {cache_file}")

    if emb is None or proto is None:
        raise RuntimeError("Embeddings or prototypes not available.")
    if user_top is None or ctx_top is None or user_mode is None:
        raise RuntimeError("Candidate pools not available.")

    # Prepare embedding index for fast cosine search
    node_list, id2idx, E, E_norm = prepare_embedding_index(emb)

    # --------------------
    # PREDICT ON ALL TEST
    # --------------------
    print("Predicting on test set...")
    preds_x, preds_y = [], []
    test_rows = test.itertuples(index=False)
    test_rows = tqdm(test_rows, total=len(test), desc="Predicting", unit="row")
    for r in test_rows:
        uid = int(r.uid); d = int(r.d); t = int(r.t) # type: ignore
        ctx = context_id(d, t)

        p = proto.get((uid, ctx))
        c = None

        if p is not None:
            cand = set(user_top.get(uid, [])) | set(ctx_top.get(ctx, []))
            if cand:
                c = cosine_best(p, cand, id2idx, E, E_norm, node_list)

        if c is None:
            c = user_mode.get(uid)

        if c is None:
            # last resort: global popular for that context, else any node
            ctx_list = ctx_top.get(ctx, [])
            c = ctx_list[0] if ctx_list else next(iter(emb.keys()))

        x, y = decode_cell(c)
        preds_x.append(x); preds_y.append(y)

    test["pred_x"] = preds_x
    test["pred_y"] = preds_y

    # --- submission behavior ---
    if fill_masked_only:
        masked_rows = (test_gt["x"] == 999) | (test_gt["y"] == 999)
        test.loc[masked_rows, "x"] = test.loc[masked_rows, "pred_x"]
        test.loc[masked_rows, "y"] = test.loc[masked_rows, "pred_y"]
    else:
        test["x"] = test["pred_x"]
        test["y"] = test["pred_y"]

    print(f"Saving predictions to {out_path}")
    test.to_csv(out_path, index=False)

    # ritorno anche masked_uids così valuti correttamente
    return test, test_gt, masked_uids

### EVALUATION

def eval_manhattan(test_pred, test_gt, masked_uids=None):
    masked_uids = set() if masked_uids is None else set(masked_uids)

    eval_rows = (
        (test_gt["x"] != 999) & (test_gt["y"] != 999) &
        (~test_gt["uid"].isin(masked_uids))
    )

    dx = (test_pred.loc[eval_rows, "pred_x"] - test_gt.loc[eval_rows, "x"]).abs()
    dy = (test_pred.loc[eval_rows, "pred_y"] - test_gt.loc[eval_rows, "y"]).abs()

    return float((dx + dy).mean())

def trajectories_from_df(df, xcol, ycol):
    # returns dict: uid -> list of (x,y) ordered by d,t
    out = {}
    for uid, g in df.sort_values(["uid","d","t"]).groupby("uid"):
        out[int(uid)] = list(zip(g[xcol].astype(int), g[ycol].astype(int)))
    return out

def make_holdout_split(train_df, holdout_days=range(46,61)):
    train_in = train_df[~train_df["d"].isin(holdout_days)].copy()
    gt_holdout = train_df[train_df["d"].isin(holdout_days)].copy()
    masked_holdout = gt_holdout.copy()
    masked_holdout["x"] = 999
    masked_holdout["y"] = 999
    return train_in, masked_holdout, gt_holdout

def align_pred_gt(pred_df, gt_df, only_days=None, masked_uids=None):
    need = {"uid", "d", "t", "x", "y"}
    if not need.issubset(pred_df.columns) or not need.issubset(gt_df.columns):
        raise ValueError(f"Servono colonne {need} in pred e gt")

    p = pred_df[["uid", "d", "t", "x", "y"]].copy()
    g = gt_df[["uid", "d", "t", "x", "y"]].copy()

    if masked_uids is not None:
        masked_uids = set(masked_uids)
        p = p[~p["uid"].isin(masked_uids)]
        g = g[~g["uid"].isin(masked_uids)]

    if only_days is not None:
        p = p[p["d"].isin(only_days)]
        g = g[g["d"].isin(only_days)]

    # togli righe senza GT
    p = p[(p["x"] != 999) & (p["y"] != 999)]
    g = g[(g["x"] != 999) & (g["y"] != 999)]

    m = p.merge(g, on=["uid", "d", "t"], suffixes=("_pred", "_gt"), how="inner")
    if m.duplicated(subset=["uid", "d", "t"]).any():
        raise ValueError("Duplicati su (uid,d,t) dopo merge. Hai righe duplicate nei CSV.")
    return m

def human_metrics(pred_df, gt_df, only_days=None, ks=(0,1,2,5), masked_uids=None):
    m = align_pred_gt(pred_df, gt_df, only_days=only_days, masked_uids=masked_uids)

    dx = (m["x_pred"].to_numpy() - m["x_gt"].to_numpy()).astype(np.int32)
    dy = (m["y_pred"].to_numpy() - m["y_gt"].to_numpy()).astype(np.int32)
    l1 = np.abs(dx) + np.abs(dy)
    l2 = np.sqrt(dx*dx + dy*dy)

    res = {
        "n_points": int(len(m)),
        "l1_mean_cells": float(l1.mean()),
        "l1_median_cells": float(np.median(l1)),
        "l2_mean_cells": float(l2.mean()),
        "l2_median_cells": float(np.median(l2)),
        "l2_mean_m": float(l2.mean() * CELL_SIZE_M),
        "l2_median_m": float(np.median(l2) * CELL_SIZE_M),
    }
    for k in ks:
        res[f"within_{k}_cells_L2"] = float((l2 <= k).mean())
    res["exact_match"] = res["within_0_cells_L2"]

    def move_rate(df_xy):
        df_xy = df_xy.sort_values(["uid","d","t"])
        x = df_xy["x"].to_numpy()
        y = df_xy["y"].to_numpy()
        uid = df_xy["uid"].to_numpy()
        same = uid[1:] == uid[:-1]
        moved = ((x[1:] != x[:-1]) | (y[1:] != y[:-1]))[same]
        return float(moved.mean()) if len(moved) else 0.0

    pred_tmp = m[["uid","d","t","x_pred","y_pred"]].rename(columns={"x_pred":"x","y_pred":"y"})
    gt_tmp   = m[["uid","d","t","x_gt","y_gt"]].rename(columns={"x_gt":"x","y_gt":"y"})
    res["move_rate_pred"] = move_rate(pred_tmp)
    res["move_rate_gt"]   = move_rate(gt_tmp)
    res["move_rate_abs_err"] = float(abs(res["move_rate_pred"] - res["move_rate_gt"]))

    pred_cells = (m["x_pred"].astype(int).astype(str) + "_" + m["y_pred"].astype(int).astype(str))
    gt_cells   = (m["x_gt"].astype(int).astype(str) + "_" + m["y_gt"].astype(int).astype(str))
    res["unique_cells_pred"] = int(pred_cells.nunique())
    res["unique_cells_gt"]   = int(gt_cells.nunique())
    res["unique_cells_ratio_pred_over_gt"] = float(res["unique_cells_pred"] / max(1, res["unique_cells_gt"]))

    return res

def eval_geobleu(pred_df, gt_df, masked_uids=None, only_days=None, n=5, beta=0.5, processes=4):
    # allinea pred/gt e filtra mascherati
    m = align_pred_gt(pred_df, gt_df, only_days=only_days, masked_uids=masked_uids)

    # 1) prova API "traj-based": from geobleu import geobleu
    gen = list(m.rename(columns={"x_pred":"x","y_pred":"y"})[["uid","d","t","x","y"]]
                .itertuples(index=False, name=None))
    ref = list(m.rename(columns={"x_gt":"x","y_gt":"y"})[["uid","d","t","x","y"]]
                .itertuples(index=False, name=None))

    return float(geobleu.calc_geobleu_bulk(gen, ref, processes=processes))

def print_metrics(title, geobleu_val, human):
    print("\n" + "="*60)
    print(title)
    print("="*60)
    if geobleu_val is None:
        print("GEO-BLEU: (skipped) installa 'geobleu' se lo vuoi")
    else:
        print(f"GEO-BLEU: {geobleu_val:.6f}")
    for k in sorted(human.keys()):
        v = human[k]
        if isinstance(v, float):
            print(f"{k:30s} {v:.6f}")
        else:
            print(f"{k:30s} {v}")

def eval_city(test_pred, test_gt, masked_uids=None, output=None):
    print("Evaluating city predictions...")
    # Manhattan solo su non-masked
    manhattan = eval_manhattan(test_pred, test_gt, masked_uids=masked_uids)

    # Human metrics classiche (solo non-masked)
    hm = human_metrics(test_pred, test_gt, masked_uids=masked_uids, ks=(0,1,2,5))

    # GeoBLEU: se installato, ok; altrimenti skip
    gb = None
    if not SKIP_GEOBLEU:
        gb = eval_geobleu(test_pred, test_gt, masked_uids=masked_uids, processes=4)

    print("\nEval results (only non-masked users):")
    print("- Manhattan: ", manhattan)
    print("- GEO-BLEU:", "(skipped)" if gb is None else f"{gb:.6f}")
    for k in sorted(hm.keys()):
        v = hm[k]
        if isinstance(v, float):
            print(f"- {k:30s} {v:.6f}")
        else:
            print(f"- {k:30s} {v}")

    # Save metrics dict
    metrics = {
        "manhattan": manhattan,
        "human": hm,
        "geobleu": gb
    }
    if output is not None:
        import json
        with open(output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {output}")
    
    return metrics

### MAIN

def main():
    for city in CITIES:
        subfolder = "/test" if city == 'T' else ""
        input_train = f"dataset/preproc{subfolder}/city_{city}_trainmerged.csv"
        input_test  = f"dataset/preproc{subfolder}/city_{city}_testmerged.csv"
        output_test = f"outputs/city_{city}_filled.csv"
        output_metrics = f"outputs/city_{city}_metrics.json"

        test_pred, test_gt, masked_uids = run_city(
            city, input_train, input_test, output_test,
            fill_masked_only=False,
            use_cache=True,
            cache_dir="cache",
            n2v_dim=64,
            n2v_walk_length=20,
            n2v_num_walks=10,
            n2v_window=10,
            n2v_workers=4,
            topN_user=200,
            topM_ctx=200
        )

        eval_city(test_pred, test_gt, masked_uids=masked_uids, output=output_metrics)
    

if __name__ == "__main__":
    main()
