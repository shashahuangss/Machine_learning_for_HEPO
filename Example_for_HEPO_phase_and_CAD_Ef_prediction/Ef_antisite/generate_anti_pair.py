import pandas as pd
import numpy as np
from itertools import product
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


# A-site and B-site composition columns
COLS_A = ['X_Y', 'X_La', 'X_Ce', 'X_Pr', 'X_Nd', 'X_Sm', 'X_Eu', 'X_Gd',
          'X_Tb', 'X_Dy', 'X_Ho', 'X_Er', 'X_Yb', 'X_Lu']
COLS_B = ['X_Ti', 'X_Zr', 'X_Hf', 'X_Sn']

# Antisite pair marker columns
COLS_A2 = ['Y_2', 'La_2', 'Ce_2', 'Pr_2', 'Nd_2', 'Sm_2', 'Eu_2', 'Gd_2',
           'Tb_2', 'Dy_2', 'Ho_2', 'Er_2', 'Yb_2', 'Lu_2']
COLS_B2 = ['Ti_2', 'Zr_2', 'Hf_2', 'Sn_2']

# Mapping from composition columns to antisite marker columns
A_TO_A2 = dict(zip(COLS_A, COLS_A2))
B_TO_B2 = dict(zip(COLS_B, COLS_B2))


def process_chunk(chunk):
    """
    For each row, generate all A-B antisite pair combinations
    from non-zero A-site and B-site elements.
    """
    results = []
    for _, row in chunk.iterrows():
        # Find non-zero elements
        non_zero_a = [c for c in COLS_A if row[c] != 0]
        non_zero_b = [c for c in COLS_B if row[c] != 0]

        combinations = list(product(non_zero_a, non_zero_b))

        for a_elem, b_elem in combinations:
            new_row = row.to_dict()
            # Set all antisite markers to 0
            for col in COLS_A2 + COLS_B2:
                new_row[col] = 0
            # Set the antisite pair
            new_row[A_TO_A2[a_elem]] = 1
            new_row[B_TO_B2[b_elem]] = 1
            results.append(new_row)

    return pd.DataFrame(results)


def main():
    print("Loading input data...")
    df = pd.read_excel("data/pyrochlore_phase_probability_0.95.xlsx")
    print(f"Input shape: {df.shape}")

    # Add empty antisite marker columns if not present
    for col in COLS_A2 + COLS_B2:
        if col not in df.columns:
            df[col] = 0

    print("Processing data with multiprocessing...")
    num_cores = multiprocessing.cpu_count()
    chunk_size = max(1, len(df) // num_cores)
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(process_chunk, chunks))

    print("Merging results...")
    result_df = pd.concat(results, ignore_index=True)
    print(f"Output shape: {result_df.shape}")

    print("Saving results...")
    output_file = "data/pyrochlore_antisite_predictions_feature.xlsx"
    result_df.to_excel(output_file, index=False)
    print(f"Results saved to '{output_file}'")


if __name__ == "__main__":
    main()
