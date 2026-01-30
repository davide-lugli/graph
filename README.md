# Human Mobility Prediction (Graph + Node2Vec)

This repository contains a possible solution to the [ACM SIGSPATIAL Cup 2025: Human Mobility Prediction Challenge](https://sigspatial2025.sigspatial.org/giscup/index.html), developed as the exam project for the Graph Analytis course @ Unimore.
The full pipeline is implemented: **train → predict → evaluate** using a **weighted directed cell-transition graph** and **Node2Vec embeddings**.

## Requirements

- Python 3.10+ (recommended)
- Create a virtual environment
- Install dependencies from `requirements.txt`
- Run Python **from the repository root directory**

## Setup

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

## Dataset Placement

The code expects datasets under:

```
dataset/
  preproc/
    city_A_trainmerged.csv
    city_A_testmerged.csv
```

For test cities whose name starts with T, the code uses:

```
dataset/
  preproc/
    test/
      city_T*_trainmerged.csv
      city_T*_testmerged.csv
```

## CSV Format

Input CSV files must include at least these columns:

- ```uid``` (integer user id)
- ```d``` (day index)
- ```t``` (time slot index)
- ```x``` (grid x coordinate)
- ```y``` (grid y coordinate)

Rows should represent user positions over time. The pipeline internally sorts by uid, d, t.

## Running

From the repo root:

```bash
python main.py
```

The cities processed are controlled by:

```python
CITIES = ['A']
```

at the beginning of the code, under the imports.

## Outputs

Predictions and evaluation metrics are written to:

```
outputs/
  city_A_filled.csv
  city_A_metrics.json
```

The pipeline also uses a cache to avoid retraining embeddings every run:

```
cache/
```

## Main Parameters

Grid configuration (in code):

```python
GRID = 200 # (200x200 grid)
CELL_SIZE_M = 500 # (500 meters per cell)
```

Disable/enable Geobleu evaluation:
```python
SKIP_GEOBLEU = True
```

Node2Vec configuration is set in the run_city(...) call inside main().
