
# Spatio-Temporal GNN: Easiest-Possible Pipeline (User Movement Prediction)

This folder contains a single Python script that:
1) Loads your CSVs (city_A/B/C/D/T_trainmerged.csv and *_testmerged.csv) in the format you described.
2) Builds a *user-as-node* graph per city using simple k-NN edges based on users' typical/home coordinates.
3) Trains a lightweight spatio-temporal GNN (T-GCN style with GRU + GraphConv) to **regress next-step (x, y)**.
4) Rolls forward to generate predictions for masked users (x=y=999 in test) in days 61–75, using real coordinates of unmasked users as context.
5) Saves per-city predictions to `./predictions/<city>_predictions.csv` with columns: uid, d, t, x_pred, y_pred.

Why this design?
- Using users as graph nodes keeps the graph small enough to train quickly on a single machine.
- Continuous (x, y) as features simplifies the model. We predict real-valued coordinates and round to grid integers at the end.
- The graph is static, learned from days 1–60 by k-NN over each user's *home* coordinate (their most frequent cell). Users who behave similarly are connected.
- During inference (days 61–75), masked users get predictions while unmasked users keep providing observed inputs. That injects real-world context into the graph.

This is **good enough for a school project**. It’s not trying to win a leaderboard.

## Requirements

- Python 3.9+
- PyTorch (CPU or CUDA)
- torch-geometric, torch-scatter, torch-sparse (matching your PyTorch/CUDA)
- networkx, numpy, pandas, scikit-learn

A minimal install (CPU) might look like:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric==2.6.1 torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
pip install pandas numpy scikit-learn networkx tqdm
```

If installing PyG wheels is painful on your machine, replace the model with the **pure-PyTorch fallback** inside the script by setting `USE_PYG = False`.

## File expectations

Put the CSVs in `./data/`:

- `city_A_trainmerged.csv`, `city_A_testmerged.csv`
- `city_B_trainmerged.csv`, `city_B_testmerged.csv`
- `city_C_trainmerged.csv`, `city_C_testmerged.csv`
- `city_D_trainmerged.csv`, `city_D_testmerged.csv`
- (Optional) `city_T_trainmerged.csv`, `city_T_testmerged.csv`

Each file must have columns:
`row_number,uid,d,t,x,y,interpolated,datetime,timestamp`

Train: days 1–60. Test: days 61–75 with last 3000 uids masked to x=y=999 (as in the challenge brief).

## How to run

```bash
python stgnn_train_infer.py --data_dir ./data --out_dir ./predictions --cities A B C D --epochs 5
```

Common flags:
- `--cities` choose any subset, e.g. `A` or `B C`
- `--epochs` training epochs (defaults to 5 for speed)
- `--history 6` how many past steps to condition on (6 = last 3 hours)
- `--knn 10` k for user-user graph
- `--hidden 64` model width
- `--use_pyg 1` try to use torch-geometric; set to 0 to use the pure-PyTorch fallback

Outputs per city: `./predictions/city_<X>_predictions.csv` with rows only for masked users, days 61–75.

## Notes

- We normalize x,y to [0,1] using city-specific min/max (0..199) then denormalize and round to int.
- For cities C and D (non-consecutive test period), the model doesn't care; it’s just rolling sequences.
- If some users appear only in test or only in train, they’re handled gracefully. Unseen users in test receive nearest-neighbor priors from population means.
- For evaluation with GEO-BLEU (if you want): see the challenge docs. Hyperparams Beta=0.5, n=5, official code linked there.

Enjoy. Or, at least, get credit.
