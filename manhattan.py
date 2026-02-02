#!/usr/bin/env python3
import csv
import heapq

PATH_TRAIN = "dataset/preproc/test/city_TA_trainmerged.csv"
PATH_PRED = "outputs/prova_1/city_TA_filled.csv"

PATH = PATH_PRED
XCOL = "x"
YCOL = "y"
EXPECTED_HEADER = ["uid", "d", "t", XCOL, YCOL]
if PATH == PATH_PRED:
    XCOL = "pred_x"
    YCOL = "pred_y"
    EXPECTED_HEADER = EXPECTED_HEADER + [XCOL, YCOL]

TOPK = 5

def parse_int_cell(s: str) -> int:
    s = s.strip()
    if s == "":
        raise ValueError("Empty x/y value")
    return int(round(float(s)))

def manhattan_step(r1, r2) -> int:
    x1 = parse_int_cell(r1[XCOL]) - parse_int_cell(r2[XCOL])
    y1 = parse_int_cell(r1[YCOL]) - parse_int_cell(r2[YCOL])
    return abs(x1) + abs(y1)

def main():
    topk = TOPK
    # min-heap of (dist, idx_prev, prev_row, next_row)
    best = []

    prev = None
    prev_idx = -1

    with open(PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        expected = EXPECTED_HEADER
        if reader.fieldnames != expected:
            raise ValueError(f"Unexpected header. Got {reader.fieldnames}, expected {expected}")

        for i, row in enumerate(reader, start=1):  # start=1 = data line number (excluding header)
            if prev is not None and row["uid"] == prev["uid"]:
                dist = manhattan_step(prev, row)
                item = (dist, prev_idx, prev, row)

                if len(best) < topk:
                    heapq.heappush(best, item)
                else:
                    # keep only the largest distances
                    if dist > best[0][0]:
                        heapq.heapreplace(best, item)

            prev = row
            prev_idx = i

    best_sorted = sorted(best, key=lambda x: x[0], reverse=True)

    if not best_sorted:
        print("No consecutive same-uid pairs found.")
        return

    print(f"Top {len(best_sorted)} consecutive same-uid jumps (Manhattan abs(dx)+abs(dy)):\n")
    for rank, (dist, idx_prev, r1, r2) in enumerate(best_sorted, start=1):
        print(f"#{rank}  dist={dist}  (lines {idx_prev} -> {idx_prev+1})  uid={r1['uid']}")
        print(f"  prev: uid={r1['uid']} d={r1['d']} t={r1['t']} x={r1[XCOL]} y={r1[YCOL]}")
        print(f"  next: uid={r2['uid']} d={r2['d']} t={r2['t']} x={r2[XCOL]} y={r2[YCOL]}")
        print()

if __name__ == "__main__":
    main()
