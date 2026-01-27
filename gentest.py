#!/usr/bin/env python3
"""
Filter a CSV like:
",uid,d,t,x,y,interpolated,datetime,timestamp"
and produce output.csv with columns: uid,d,t,x,y
keeping only uids in [1..N].
"""

import csv
from pathlib import Path


IN_TRAIN = "dataset/preproc/city_A_trainmerged.csv"
IN_TEST = "dataset/preproc/city_A_testmerged.csv"
OUT_TRAIN = "dataset/preproc/test/city_TB_trainmerged.csv"
OUT_TEST = "dataset/preproc/test/city_TB_testmerged.csv"


def normalize_uid(value: str) -> int | None:
    """
    Convert uid field to int if possible, else None.
    Accepts strings like '123', '123.0', ' 123 '.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        try:
            f = float(s)
            if f.is_integer():
                return int(f)
        except ValueError:
            return None
    return None

def filter_csv(input_path: Path, output_path: Path, max_uid: int) -> None:
    kept = 0
    total = 0

    with input_path.open("r", newline="", encoding="utf-8") as f_in, \
        output_path.open("w", newline="", encoding="utf-8") as f_out:

        reader = csv.DictReader(f_in)

        required = ["uid", "d", "t", "x", "y"]
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"Missing required columns in input: {missing}. Found: {reader.fieldnames}")

        writer = csv.DictWriter(f_out, fieldnames=required)
        writer.writeheader()

        for row in reader:
            total += 1
            uid = normalize_uid(str(row.get("uid")))
            if uid is None:
                continue
            if 1 <= uid <= max_uid:
                writer.writerow({k: row.get(k, "") for k in required})
                kept += 1

    print(f"Done. Read {total} rows, wrote {kept} rows to {output_path}")


def main() -> int:
    in_train_path = Path(IN_TRAIN)
    out_train_path = Path(OUT_TRAIN)
    in_test_path = Path(IN_TEST)
    out_test_path = Path(OUT_TEST)

    print(f"Filtering train data: {in_train_path} -> {out_train_path}")
    filter_csv(in_train_path, out_train_path, max_uid=10000)
    print(f"Filtering test data: {in_test_path} -> {out_test_path}")
    filter_csv(in_test_path, out_test_path, max_uid=10000)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
