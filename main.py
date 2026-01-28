import os
import gzip
import pickle
import hashlib
import geobleu
import pandas as pd
import numpy as np
from pecanpy import pecanpy as p2v
from collections import defaultdict, Counter
from tqdm import tqdm

CITIES = ['TB']
SKIP_GEOBLEU = True
GRID = 200
CELL_SIZE_M = 500  # 500m per cella

### HELPERS ###

'''
Converte coordinate (x, y) della griglia (1..200) in un id intero unico.
Mappatura "row-major": tutte le celle della riga x=1, poi x=2, ecc.

Esempio con GRID=200:
(1,1) -> 0
(1,2) -> 1
...
(1,200) -> 199
(2,1) -> 200
'''
def cell_id(x, y):
    # x,y in [1..200]
    return (x - 1) * GRID + (y - 1)

'''
Inversa di cell_id: da id cella a (x, y)
'''
def decode_cell(c):
    x = c // GRID + 1
    y = c % GRID + 1
    return x, y

'''
Trasforma un "day index" d (1..75 nel challenge) nel giorno della settimana 0..6
'''
def weekday_of(d):
    return (d - 1) % 7

'''
Crea un id contesto temporale combinando:
- weekday (0..6) * 48
- timeslot t (0..47)
Serve per creare nodi/feature "spazio-tempo"
'''
def context_id(d, t):
    return weekday_of(d) * 48 + t  # 0..335

### CACHING ###
### Usati per non ricalcolare ogni volta dati intermedi (embeddings, prototypes, ecc.) se non cambiano i dati o i parametri

'''
Fingerprint veloce:
- size
- mtime
- hash primi+ultimi blocchi

Serve a capire se un file è "cambiato abbastanza" da invalidare la cache.
'''
def file_fingerprint(path, block_size=1 << 20):
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

'''
Genera nome file di cache
'''
def cache_path_for(city, train_path, cache_dir="cache", tag="n2v_proto"):
    os.makedirs(cache_dir, exist_ok=True)
    fp = file_fingerprint(train_path)
    return os.path.join(cache_dir, f"{tag}_city_{city}_{fp}.pkl.gz")

'''
Salva cache gzip+pickle
'''
def save_cache(path, payload):
    with gzip.open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL) # type: ignore

'''
Carica cache gzip+pickle
'''
def load_cache(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f) # type: ignore

### PREDICTION

'''
Estrae le top-N celle più visitate per ogni utente nel train set
'''
def top_cells_by_user(train_df, topN=200):
    out = {}
    # Raggruppa tutto il train per utente
    for uid, g in train_df.groupby("uid"):
        # Converte ogni (x,y) in cell_id lineare (0..39999)
        cells = [cell_id(int(x), int(y)) for x, y in zip(g["x"], g["y"])]
        # Conta frequenze delle celle per quell'utente e prende le topN più frequenti
        out[int(uid)] = [c for c, _ in Counter(cells).most_common(topN)]
    return out

'''
Estrae le top-M celle più visitate per ogni contesto (weekday+timeslot) nel train set
'''
def top_cells_by_context(train_df, topM=200):
    out = {}
    # Lista di celle visitate in quel contesto (weekday+slot)
    ctx_cells = defaultdict(list)
    for r in train_df.itertuples(index=False):
        ctx = context_id(int(r.d), int(r.t))
        # Aggiunge cella visitata in quel contesto
        ctx_cells[ctx].append(cell_id(int(r.x), int(r.y)))
    # Per ogni contesto, prende le celle più popolari globalmente
    for ctx, cells in ctx_cells.items():
        out[ctx] = [c for c, _ in Counter(cells).most_common(topM)]
    return out

'''
Prepara strutture per cosine_best fast:
- node_list: array dei node ids (celle)
- id2idx: mapping cell_id -> row index
- E: embedding matrix (N, dim)
- E_norm: norma per ogni embedding (N,)
'''
def prepare_embedding_index(emb: dict):
    node_list = np.array(list(emb.keys()), dtype=np.int32)
    id2idx = {int(n): i for i, n in enumerate(node_list)}
    E = np.stack([emb[int(n)] for n in node_list]).astype(np.float32)
    E_norm = np.linalg.norm(E, axis=1).astype(np.float32) + 1e-9
    return node_list, id2idx, E, E_norm

'''
Ritorna il candidate (cell_id) con massima cosine similarity con proto_vec
'''
def cosine_best(proto_vec, candidates, id2idx, E, E_norm, node_list):
    if proto_vec is None:
        return None

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

    scores = (E[cand_idx] @ pv) / (E_norm[cand_idx] * pv_norm)

    best_local = int(np.argmax(scores))
    best_idx = int(cand_idx[best_local])
    return int(node_list[best_idx])

'''
Trasforma le traiettorie in un grafo di transizioni cella→cella, pesato per frequenza
'''
def build_edgelist_file(train_df, edg_path):
    # Le righe sono osservazioni sequenziali. Ordina per user e tempo.
    train_df = train_df.sort_values(["uid", "d", "t"])
    uid = train_df["uid"].to_numpy(dtype=np.int32)
    x = train_df["x"].to_numpy(dtype=np.int16)
    y = train_df["y"].to_numpy(dtype=np.int16)

    # cell_id vettorializzato (equivalente a cell_id(x,y) ma senza loop python)
    cells = (x.astype(np.int32) - 1) * GRID + (y.astype(np.int32) - 1)  # 0..39999

    # Transizione solo se la riga i e i+1 sono dello stesso utente
    same_user = uid[1:] == uid[:-1]
    src = cells[:-1][same_user]
    dst = cells[1:][same_user]

    # Rimuove self-loop (stessa cella -> stessa cella)
    m = src != dst
    src = src[m].astype(np.int32)
    dst = dst[m].astype(np.int32)

    # Comprime la coppia (src,dst) in un singolo int64 per contare con np.unique
    M = GRID * GRID  # 40000
    key = src.astype(np.int64) * M + dst.astype(np.int64)

    # uniq = chiavi uniche, cnt = conteggi (frequenze transizioni)
    uniq, cnt = np.unique(key, return_counts=True)
    
    # Decomprime di nuovo in src,dst
    src_u = (uniq // M).astype(np.int32)
    dst_u = (uniq % M).astype(np.int32)
    w_u = cnt.astype(np.float32)

    # Scrive edgelist: "u \t v \t weight"
    os.makedirs(os.path.dirname(edg_path) or ".", exist_ok=True)
    with open(edg_path, "w") as f:
        for u, v, w in zip(src_u, dst_u, w_u):
            f.write(f"{u}\t{v}\t{w}\n")

    return edg_path

'''
Costruisce i prototipi medi per ogni (uid, context)
In pratica perogni utente e contesto trova il centroide delle celle visitate
'''
def build_prototypes(train_df, emb):
    # buckets[(uid, ctx)] = lista di embedding vettori
    buckets = defaultdict(list)

    for r in train_df.itertuples(index=False):
        c = cell_id(int(r.x), int(r.y))
        # Se quella cella non ha embedding (es. nodo isolato o filtrato), skip
        if c not in emb:
            continue
        ctx = context_id(int(r.d), int(r.t))
        buckets[(int(r.uid), ctx)].append(emb[c])

    # proto[(uid, ctx)] = media dei vettori nel bucket
    proto = {}
    for k, vecs in buckets.items():
        proto[k] = np.mean(vecs, axis=0)
    return proto

'''
Costruisce le top-K transizioni per ogni cella
Ritorna dict: prev_cell -> [next_cell1, next_cell2, ...] (ordinati per frequenza)
'''
def build_topk_transitions(train_df, K=30): 
    df = train_df.sort_values(["uid", "d", "t"])

    uid = df["uid"].to_numpy(dtype=np.int32)
    x = df["x"].to_numpy(dtype=np.int16)
    y = df["y"].to_numpy(dtype=np.int16)
    
    cells = (x.astype(np.int32) - 1) * GRID + (y.astype(np.int32) - 1)
    
    same_user = uid[1:] == uid[:-1]
    src = cells[:-1][same_user]
    dst = cells[1:][same_user] # drop self loops
    
    m = src != dst
    src = src[m].astype(np.int32)
    dst = dst[m].astype(np.int32)
    
    M = GRID * GRID # 40000
    
    key = src.astype(np.int64) * M + dst.astype(np.int64)
    uniq, cnt = np.unique(key, return_counts=True)
    src_u = (uniq // M).astype(np.int32)
    dst_u = (uniq % M).astype(np.int32)
    w_u = cnt.astype(np.int32) # raggruppa per src e tieni top-K
    
    topk = {}
    order = np.argsort(src_u, kind="mergesort")
    
    src_u, dst_u, w_u = src_u[order], dst_u[order], w_u[order]
    i = 0
    n = len(src_u)
    
    while i < n:
        u = int(src_u[i])
        j = i
    
        while j < n and src_u[j] == src_u[i]:
            j += 1 # per questo u, prendi top-K per weight
            idx = np.argsort(w_u[i:j])[::-1]
            best = dst_u[i:j][idx][:K]
            topk[u] = best.astype(np.int32).tolist()
            i = j
    
    return topk

'''
Training di Node2Vec usando PecanPy a partire da un file edgelist pesato (cella → cella con conteggio transizioni)
In pratica:
1. Legge il grafo (diretto e pesato) creato con build_edgelist_file
2. A seconda del mode prepara o calcola “al volo” le probabilità di transizione biased di Node2Vec (controllate da p e q)
3. Genera random-walks sul grafo e allena uno Skip-gram stile word2vec su quelle sequenze
4. Restituisce un dizionario emb[cell_id] = vettore (float32) di dimensione dim
'''
def train_node2vec_pecanpy(edg_path, dim=64, walk_length=20, num_walks=10, window=10, workers=4, p=1.0, q=1.0, mode="SparseOTF", directed=True, extend=False):
    # mode: "PreComp", "SparseOTF", "DenseOTF"
    # In PecanPy esistono più backend/strategie per gestire le probabilità di transizione:
    # - PreComp: pre-calcola TUTTE le transition probs (più RAM, più veloce dopo)
    # - SparseOTF: calcola on-the-fly in modo ottimizzato per grafi sparsi (di solito la scelta giusta)
    # - DenseOTF: on-the-fly ma pensato per grafi densi
    cls = {"PreComp": p2v.PreComp, "SparseOTF": p2v.SparseOTF, "DenseOTF": p2v.DenseOTF}[mode]

    # Crea l'oggetto grafo Node2Vec di PecanPy
    # p e q iperparametri di Node2Vec che controllano il bias delle random-walk:
    # - p (return parameter): quanto penalizza tornare indietro al nodo precedente
    # - q (in-out parameter): quanto favorisca esplorare lontano vs restare vicino
    #
    # workers: parallelismo
    # extend: opzione PecanPy per gestire nodi isolati/estensioni
    g = cls(p=p, q=q, workers=workers, verbose=True, extend=extend)

    # Legge dal file edgelist che è "u \t v \t w" (peso = frequenza transizioni)
    # weighted=True: usa w come peso arco (influenza la probabilità di camminata)
    # directed=directed: grafo diretto se True (cella->cella), non simmetrizza
    g.read_edg(edg_path, weighted=True, directed=directed)

    # Pre-calcola probabilità di transizione (solo mode PreComp)
    if mode == "PreComp":
        g.preprocess_transition_probs()

    # embed() è la parte "word2vec-like":
    # 1) genera random walks sul grafo (num_walks per nodo, lunghezza walk_length)
    # 2) allena Skip-gram con finestra window_size (come contesto nelle sequenze)
    # epochs=1: una sola epoca di training dello skip-gram (spesso basta)
    emd = g.embed(
        dim=dim,
        num_walks=num_walks,
        walk_length=walk_length,
        window_size=window,
        epochs=1,
        verbose=True
    )

    # emd è una matrice (N, dim): un embedding per nodo
    # Le righe di emd sono allineate con g.nodes
    # g.nodes sono gli id dei nodi come stringhe
    nodes = g.nodes

    # Converte in un dizionario Python:
    # emb[cell_id_int] = embedding (float32)
    emb = {int(nodes[i]): emd[i].astype(np.float32) for i in range(len(nodes))}
    
    return emb

'''
Pipeline di train e prediction per una città:
1. Genera edgelist, prototipi, candidate pools
2. Addestra Node2Vec su TUTTI gli utenti del train set
3. Fa prediction su TUTTI gli utenti del test set
4. Ritorna DataFrame test con predizioni e lista masked_uids
'''
def run_city(city, city_train_path, city_test_path, out_path, use_cache=True, cache_dir="cache", 
    n2v_dim=64, 
    n2v_walk_length=20, 
    n2v_num_walks=10, 
    n2v_window=10, 
    n2v_workers=4, 
    n2v_p=1.0,
    n2v_q=2.0,
    n2v_mode="SparseOTF",
    n2v_directed=True,
    n2v_extend=False,
    topN_user=100, 
    topM_ctx=100,
    topK_trans=30
):
    print(f"Running city pipeline for CITY={city}")

    # Legge i CSV di train e test
    train = pd.read_csv(city_train_path)
    test  = pd.read_csv(city_test_path)

    # Trova gli utenti "masked": cioè quelli che nel test hanno almeno una riga con x=999 o y=999
    masked_uids = set(test.loc[(test["x"] == 999) | (test["y"] == 999), "uid"].unique())

    # --------------------
    # TRAIN ON ALL USERS AND CACHE
    # --------------------

    # Costruisce un path cache unico basato su city + fingerprint del train file
    # Se cambia il train file, cambia la cache (nome diverso)
    cache_file = cache_path_for(city, city_train_path, cache_dir=cache_dir)
    # Serve per capire se la cache è ancora valida anche se il file è uguale
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
        "topK_trans": topK_trans
    }

    # Inizializza a None: quando carica cache o ricostruisce li riempie
    emb = None
    proto = None
    topk_transitions = None
    user_top = None
    ctx_top = None
    user_mode = None

    # Se caching abilitato e file cache esiste
    if use_cache and os.path.exists(cache_file):
        payload = load_cache(cache_file)

        # Controlla se cache_key combacia, la cache è valida, se no ricostruisci
        if payload.get("cache_key") == cache_key:
            emb = payload["emb"]
            proto = payload["proto"]
            topk_transitions = payload["topk_transitions"]
            user_top = payload["user_top"]
            ctx_top = payload["ctx_top"]
            user_mode = payload["user_mode"]
            print(f"Loaded cache: {cache_file}")
        else:
            print("Cache found but params mismatch, rebuilding...")
            payload = None
    else:
        payload = None

    # Se non ha cache valida, ricostruisce tutto
    if payload is None:
        # Uso tutti gli utenti per il training
        train_used = train

        print("Building graph and embeddings...")

        # Costruisce file edgelist pesato dalle transizioni (uid,d,t ordinato) e lo mette in cache_dir (non nel fingerprint cache file)
        edg_path = os.path.join(cache_dir, f"city_{city}_graph.edg")
        build_edgelist_file(train_used, edg_path)

        # Allena embedding Node2Vec sulle celle del grafo
        emb = train_node2vec_pecanpy(
            edg_path,
            dim=n2v_dim,
            walk_length=n2v_walk_length,
            num_walks=n2v_num_walks,
            window=n2v_window,
            workers=n2v_workers,
            p=n2v_p,
            q=n2v_q,
            mode=n2v_mode,
            directed=n2v_directed,
            extend=n2v_extend
        )

        print("Graph and embeddings built.")
        print("Building prototypes and candidate pools...")

        # Prototipi embedding medi per bucket (uid, context_id(d,t))
        # Serve per avere una "firma" di dove sta l'utente in quel contesto
        proto = build_prototypes(train_used, emb)

        # Transizioni top-K globali: prev_cell -> top-K next cells
        topk_transitions = build_topk_transitions(train_used, K=topK_trans)

        # Pool candidati fallback: celle più frequenti per utente e per contesto
        user_top = top_cells_by_user(train_used, topN=topN_user)
        ctx_top  = top_cells_by_context(train_used, topM=topM_ctx)

        # Fallback: per ogni utente sceglie la cella più frequente in assoluto
        # Serve quando: non ha prototipo (bucket vuoto) e non ha embedding per candidati
        user_mode = {}
        for uid, g in train_used.groupby("uid"):
            cells = [cell_id(int(x), int(y)) for x, y in zip(g["x"], g["y"])]
            user_mode[int(uid)] = Counter(cells).most_common(1)[0][0] # type: ignore

        print("Prototypes and candidate pools built.")

        # Salva cache compressa: così la prossima run evita Node2Vec (parte più costosa)
        if use_cache:
            save_cache(cache_file, {
                "cache_key": cache_key,
                "emb": emb,
                "proto": proto,
                "topk_transitions": topk_transitions,
                "user_top": user_top,
                "ctx_top": ctx_top,
                "user_mode": user_mode,
            })
            print(f"Saved cache: {cache_file}")

    # Sanity check: se qualcosa è rimasto None, la pipeline è rotta e abortisce
    if emb is None or proto is None:
        raise RuntimeError("Embeddings or prototypes not available.")
    if user_top is None or ctx_top is None or user_mode is None or topk_transitions is None:
        raise RuntimeError("Candidate pools not available.")

    # Prepara strutture NumPy per cosine_best: node_list, id2idx, E, E_norm
    # Serve per poter fare cosine su tanti candidati in prediction senza loop pesanti
    node_list, id2idx, E, E_norm = prepare_embedding_index(emb)

    # --------------------
    # PREDICT ON ALL TEST
    # --------------------

    # Si assicura che il test sia in ordine temporale per utente
    test = test.sort_values(["uid", "d", "t"]).reset_index(drop=True)

    print("Predicting on test set...")

    preds_x, preds_y = [], []
    
    # Cella precedente per ciascun uid, aggiornato man mano (usa GT se disponibile, altrimenti pred)
    prev_cell = {}

    # Itera riga per riga sul test set
    test_rows = tqdm(test.itertuples(index=False), total=len(test), desc="Predicting", unit="row")
    for i, r in enumerate(test_rows):
        # Estrae i campi principali della riga
        # uid: utente, d: giorno, t: timeslot
        uid = int(r.uid); d = int(r.d); t = int(r.t) # type: ignore

        # Contesto temporale: weekday(d)*48 + t
        ctx = context_id(d, t)

        # p = prototipo embedding medio per (uid, ctx)
        # Se non esiste, vuol dire che nel train non ha mai visto quell'utente in quel contesto
        p = proto.get((uid, ctx))

        # Cella precedente per questo uid
        prev = prev_cell.get(uid)

        # c = cella predetta (cell_id)
        c = None

        # Costruisce candidati:
        # - user_top[uid]: celle più frequentate da quell'utente (prior personale)
        # - ctx_top[ctx]: celle più popolari in quel contesto (prior globale)
        # Elimina duplicati con set()
        cand = set(user_top.get(uid, [])) | set(ctx_top.get(ctx, []))

        # Aggiungi candidati dalle transizioni della cella precedente
        if prev is not None:
            nxt = topk_transitions.get(prev)
            if nxt:
                cand.update(nxt)  # restringe molto verso mosse realistiche

        # Prima scelta: se ha un prototipo cerca il candidato con massima cosine similarity con esso
        if p is not None and cand:
            c = cosine_best(p, cand, id2idx, E, E_norm, node_list)

        # Seconda scelta: se non ha prototipo o cosine_best fallisce, ma hai prev, usa la transizione più frequente
        if c is None and prev is not None:
            nxt = topk_transitions.get(prev)
            if nxt:
                c = int(nxt[0])

        # Terza scelta: usa la cella più frequente in assoluto per quell'utente
        if c is None:
            c = user_mode.get(uid)

        # Quarta scelta: se ancora nulla (es. utente nuovo), usa la cella più popolare in quel contesto
        if c is None:
            ctx_list = ctx_top.get(ctx, [])
            
            # Se non esiste la cella più popolare per quel contesto, prende una qualsiasi cella dall'embedding
            c = ctx_list[0] if ctx_list else next(iter(emb.keys()))

        # Decodifica cell_id in (x,y)
        x, y = decode_cell(c)

        # Salva predizioni per questa riga
        preds_x.append(x)
        preds_y.append(y)

        # Aggiorna cella precedente per questo uid usando la predizione
        prev_cell[uid] = c

    # Aggiunge colonne predizioni al DataFrame test
    test["pred_x"] = preds_x
    test["pred_y"] = preds_y

    # --------------------
    # SUBMISSION
    # --------------------

    print(f"Saving predictions to {out_path}")

    # Salva risultati precition in CSV
    test.to_csv(out_path, index=False)

    # Ritorna test (con ground truth "x", "y" e prediction "pred_x", "pred_y")
    # Ritorna anche eventuale lista di masked_uids in modo da escluderli da evaluation
    return test, masked_uids

### EVALUATION

'''
Ritorna un dataframe filtrato per evaluation:
- Esclude gli utenti in masked_uids (se fornito)
- Filtra solo alcuni giorni (se only_days fornito)
- Droppa righe con pred_x/pred_y mancanti
'''
def filter_eval_rows(test_df: pd.DataFrame, masked_uids=None, only_days=None) -> pd.DataFrame:
    
    df = test_df

    if masked_uids is not None:
        mu = set(masked_uids)
        df = df[~df["uid"].isin(mu)]

    if only_days is not None:
        df = df[df["d"].isin(set(only_days))]

    current_df_len = len(df)
    df = df.dropna(subset=["pred_x", "pred_y", "x", "y"])
    dropped = current_df_len - len(df)
    if dropped > 0:
        print(f"[WARNING] Dropped {dropped} rows with missing predictions or ground truth.")

    return df

'''
Calcola rate di movimento tra timestep consecutivi per lo stesso utente.
Movimento = cambia x o y rispetto alla riga precedente dello stesso uid.
'''
def move_rate(df_xy: pd.DataFrame) -> float:
    tmp = df_xy.sort_values(["uid", "d", "t"])
    x = tmp["x"].to_numpy()
    y = tmp["y"].to_numpy()
    uid = tmp["uid"].to_numpy()

    same = uid[1:] == uid[:-1]
    moved = ((x[1:] != x[:-1]) | (y[1:] != y[:-1]))[same]
    return float(moved.mean()) if len(moved) else 0.0

'''
Calcola mean Manhattan distance (espressa in CELLS) tra (pred_x,pred_y) e (x,y)
'''
def eval_manhattan(df):
    dx = (df["pred_x"] - df["x"]).abs()
    dy = (df["pred_y"] - df["y"]).abs()

    return float((dx + dy).mean()) if len(df) else float("nan")

'''
Calcola metriche "human-friendly" su celle/metri e statistiche di movimento
- L1 e L2 mean/median in celle
- L2 mean/median in metri
- Percentuali within_k_cells_L2 per k in ks
- Exact match (within 0 cells L2)
- Move rate pred/gt/abs error
- Unique cells pred/gt/ratio
'''
def human_metrics(df, ks=(0,1,2,5)):
    # Se non ci sono punti valutabili, ritorna metriche vuote
    if len(df) == 0:
        return {
            "n_points": 0,
            "l1_mean_cells": float("nan"),
            "l1_median_cells": float("nan"),
            "l2_mean_cells": float("nan"),
            "l2_median_cells": float("nan"),
            "l2_mean_m": float("nan"),
            "l2_median_m": float("nan"),
            **{f"within_{k}_cells_L2": float("nan") for k in ks},
            "exact_match": float("nan"),
            "move_rate_pred": float("nan"),
            "move_rate_gt": float("nan"),
            "move_rate_abs_err": float("nan"),
            "unique_cells_pred": 0,
            "unique_cells_gt": 0,
            "unique_cells_ratio_pred_over_gt": float("nan"),
        }

    dx = (df["pred_x"].to_numpy() - df["x"].to_numpy()).astype(np.int32)
    dy = (df["pred_y"].to_numpy() - df["y"].to_numpy()).astype(np.int32)

    l1 = np.abs(dx) + np.abs(dy)
    l2 = np.sqrt(dx * dx + dy * dy)

    res = {
        "n_points": int(len(df)),
        "l1_mean_cells": float(l1.mean()),
        "l1_median_cells": float(np.median(l1)),
        "l2_mean_cells": float(l2.mean()),
        "l2_median_cells": float(np.median(l2)),
        "l2_mean_meters": float(l2.mean() * CELL_SIZE_M),
        "l2_median_meters": float(np.median(l2) * CELL_SIZE_M),
    }

    for k in ks:
        res[f"within_{k}_cells_L2"] = float((l2 <= k).mean())
    res["exact_match"] = res["within_0_cells_L2"]

    pred_tmp = df[["uid", "d", "t", "pred_x", "pred_y"]].rename(columns={"pred_x": "x", "pred_y": "y"})
    gt_tmp   = df[["uid", "d", "t", "x", "y"]]

    res["move_rate_pred"] = move_rate(pred_tmp)
    res["move_rate_gt"]   = move_rate(gt_tmp)
    res["move_rate_abs_err"] = float(abs(res["move_rate_pred"] - res["move_rate_gt"]))

    pred_cells = df["pred_x"].astype(int).astype(str) + "_" + df["pred_y"].astype(int).astype(str)
    gt_cells   = df["x"].astype(int).astype(str)      + "_" + df["y"].astype(int).astype(str)

    res["unique_cells_pred"] = int(pred_cells.nunique())
    res["unique_cells_gt"]   = int(gt_cells.nunique())
    res["unique_cells_ratio_pred_over_gt"] = float(res["unique_cells_pred"] / max(1, res["unique_cells_gt"]))

    return res

'''
Calcola GEO-BLEU su traiettorie
Costruisce gen/ref direttamente da:
- gen: (uid,d,t,pred_x,pred_y)
- ref: (uid,d,t,x,y)
'''
def eval_geobleu(df, processes=4):
    # Formato richiesto dalla tua chiamata precedente: tuple (uid,d,t,x,y)
    gen = list(
        df.rename(columns={"pred_x": "x", "pred_y": "y"})[["uid", "d", "t", "x", "y"]].itertuples(index=False, name=None)
    )
    ref = list(
        df[["uid", "d", "t", "x", "y"]].itertuples(index=False, name=None)
    )

    return float(geobleu.calc_geobleu_bulk(gen, ref, processes=processes))

'''
Pipeline di evaluation per una città:
- Calcola e stampa metriche Manhattan, GEO-BLEU, Human Metrics
'''
def eval_city(test_df, masked_uids=None, output=None):
    print("Evaluating city predictions...")

    # Filtra righe per evaluation (rimuove masked users, droppa righe senza predizioni)
    eval_df = filter_eval_rows(test_df, masked_uids=masked_uids)

    # Manhattan
    manhattan = eval_manhattan(eval_df)

    # Human metrics classiche
    hm = human_metrics(eval_df, ks=(0,1,2,5))

    # GeoBLEU (se abilitato)
    gb = None
    if not SKIP_GEOBLEU:
        gb = eval_geobleu(eval_df, processes=4)

    print("\nEval results (only non-masked users):")
    print("- Manhattan: ", manhattan)
    print("- GEO-BLEU:", "(skipped)" if gb is None else f"{gb:.6f}")
    for k in sorted(hm.keys()):
        v = hm[k]
        if isinstance(v, float):
            print(f"- {k:30s} {v:.6f}")
        else:
            print(f"- {k:30s} {v}")

    # Salva metriche su file JSON se richiesto
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
    # Per ogni città, esegue pipeline completa di train, prediction e evaluation
    for city in CITIES:
        subfolder = "/test" if city.startswith('T') else ""
        input_train = f"dataset/preproc{subfolder}/city_{city}_trainmerged.csv"
        input_test  = f"dataset/preproc{subfolder}/city_{city}_testmerged.csv"
        output_test = f"outputs/city_{city}_filled.csv"
        output_metrics = f"outputs/city_{city}_metrics.json"

        # Train e inference
        test_df, masked_uids = run_city(
            city, input_train, input_test, output_test,
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

        # Evaluation
        eval_city(test_df, masked_uids=masked_uids, output=output_metrics)
    
if __name__ == "__main__":
    main()
