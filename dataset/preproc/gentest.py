import pandas as pd
import os

def filter_large_csv(input_path, output_path, uid_max=2000, chunksize=100000):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Prepare the output file
    with open(output_path, 'w') as f_out:
        header_written = False

        for chunk in pd.read_csv(input_path, chunksize=chunksize):
            filtered_chunk = chunk[chunk['uid'].between(0, uid_max)]

            # Write header only once
            filtered_chunk.to_csv(
                f_out,
                index=False,
                header=not header_written,
                mode='a'
            )
            header_written = True

def replace_last_uids_xy(input_file, output_file, uid_col='uid', x_col='x', y_col='y', last_n=500, chunksize=100000):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Step 1: Find the max UID
    uid_max = None
    for chunk in pd.read_csv(input_file, usecols=[uid_col], chunksize=chunksize):
        m = chunk[uid_col].max()
        uid_max = m if uid_max is None else max(uid_max, m)

    if uid_max is None:
        print("No data found in file.")
        return

    uid_threshold = uid_max - last_n + 1  # Modify rows where uid >= this

    # Step 2: Process and write to new file
    with open(output_file, 'w') as f_out:
        header_written = False
        for chunk in pd.read_csv(input_file, chunksize=chunksize):
            mask = chunk[uid_col] >= uid_threshold
            chunk.loc[mask, [x_col, y_col]] = 999.0
            chunk.to_csv(f_out, index=False, header=not header_written, mode='a')
            header_written = True

# File paths
train_in = "city_A_trainmerged.csv"
test_in  = "city_A_testmerged.csv"
train_out = "test/city_T_trainmerged.csv"
test_out  = "test/city_T_testmerged_original.csv"
test_out_masked  = "test/city_T_testmerged.csv"

# Run filtering
filter_large_csv(train_in, train_out)
filter_large_csv(test_in, test_out)
replace_last_uids_xy(test_out, test_out_masked)

print("Done. Filtered files saved in dataset/preproc/test/")
