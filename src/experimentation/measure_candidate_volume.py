
import os
import statistics

import pandas as pd

from src.blocking.candidate_generation import (
    build_block_index,
    get_candidates,
)


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE1_PATH = "./data/dataset/train/train_source1.tsv"
SOURCE2_PATH = "./data/dataset/train/train_source2.tsv"
SOURCE3_PATH = "./data/dataset/train/train_source3.tsv"

SAMPLE_SIZE = 10_000
RANDOM_SEED = 42


# ============================================================
# HELPERS
# ============================================================

def percentile(values, percentile):
    """
    Calculate a percentile without requiring numpy.
    """

    if not values:
        return 0

    values = sorted(values)

    index = (len(values) - 1) * percentile / 100
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)

    if lower == upper:
        return values[lower]

    weight = index - lower

    return (
        values[lower] * (1 - weight)
        + values[upper] * weight
    )


def print_statistics(label, values):
    """
    Print candidate-count statistics.
    """

    if not values:
        print(f"\n{label}")
        print("-" * 60)
        print("No values.")
        return

    print(f"\n{label}")
    print("-" * 60)

    print(f"Records measured : {len(values):,}")
    print(f"Total candidates : {sum(values):,}")
    print(f"Average          : {statistics.mean(values):,.2f}")
    print(f"Median           : {statistics.median(values):,.2f}")
    print(f"P95              : {percentile(values, 95):,.2f}")
    print(f"P99              : {percentile(values, 99):,.2f}")
    print(f"Maximum          : {max(values):,}")
    print(f"Minimum          : {min(values):,}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("          CANDIDATE VOLUME MEASUREMENT")
    print("=" * 70)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    for path in [
        SOURCE1_PATH,
        SOURCE2_PATH,
        SOURCE3_PATH,
    ]:

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    # --------------------------------------------------------
    # Load Source 1
    # --------------------------------------------------------

    print("\nLoading Source 1...")

    source1_df = pd.read_csv(
        SOURCE1_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print(
        f"Source 1 records: "
        f"{len(source1_df):,}"
    )

    # --------------------------------------------------------
    # Sample Source 1
    # --------------------------------------------------------

    sample_size = min(
        SAMPLE_SIZE,
        len(source1_df)
    )

    source1_sample = source1_df.sample(
        n=sample_size,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)

    print(
        f"Source 1 sample: "
        f"{len(source1_sample):,}"
    )

    # --------------------------------------------------------
    # Load Source 2
    # --------------------------------------------------------

    print("\nLoading Source 2...")

    source2_df = pd.read_csv(
        SOURCE2_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print(
        f"Source 2 records: "
        f"{len(source2_df):,}"
    )

    # --------------------------------------------------------
    # Build Source 2 index
    # --------------------------------------------------------

    print("\nBuilding Source 2 block index...")

    source2_index = build_block_index(
        source2_df
    )

    print(
        f"Source 2 block keys: "
        f"{len(source2_index):,}"
    )

    # --------------------------------------------------------
    # Load Source 3
    # --------------------------------------------------------

    print("\nLoading Source 3...")

    source3_df = pd.read_csv(
        SOURCE3_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print(
        f"Source 3 records: "
        f"{len(source3_df):,}"
    )

    # --------------------------------------------------------
    # Build Source 3 index
    # --------------------------------------------------------

    print("\nBuilding Source 3 block index...")

    source3_index = build_block_index(
        source3_df
    )

    print(
        f"Source 3 block keys: "
        f"{len(source3_index):,}"
    )

    # --------------------------------------------------------
    # Measure candidate volume
    # --------------------------------------------------------

    print("\nMeasuring candidate volume...")

    s2_counts = []
    s3_counts = []
    combined_counts = []

    s2_zero = 0
    s3_zero = 0
    combined_zero = 0

    for _, source1_record in source1_sample.iterrows():

        s2_candidates = get_candidates(
            source1_record,
            source2_index
        )

        s3_candidates = get_candidates(
            source1_record,
            source3_index
        )

        s2_count = len(s2_candidates)
        s3_count = len(s3_candidates)
        combined_count = s2_count + s3_count

        s2_counts.append(s2_count)
        s3_counts.append(s3_count)
        combined_counts.append(combined_count)

        if s2_count == 0:
            s2_zero += 1

        if s3_count == 0:
            s3_zero += 1

        if combined_count == 0:
            combined_zero += 1

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("                    RESULTS")
    print("=" * 70)

    print_statistics(
        "SOURCE 2 CANDIDATES",
        s2_counts
    )

    print_statistics(
        "SOURCE 3 CANDIDATES",
        s3_counts
    )

    print_statistics(
        "COMBINED CANDIDATES",
        combined_counts
    )

    # --------------------------------------------------------
    # Zero-candidate statistics
    # --------------------------------------------------------

    print("\nZERO-CANDIDATE RECORDS")
    print("-" * 60)

    print(
        f"Source 2 zero candidates : "
        f"{s2_zero:,} "
        f"({s2_zero / sample_size * 100:.2f}%)"
    )

    print(
        f"Source 3 zero candidates : "
        f"{s3_zero:,} "
        f"({s3_zero / sample_size * 100:.2f}%)"
    )

    print(
        f"Combined zero candidates : "
        f"{combined_zero:,} "
        f"({combined_zero / sample_size * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # Extrapolation
    # --------------------------------------------------------

    average_combined = statistics.mean(
        combined_counts
    )

    total_source1 = len(source1_df)

    estimated_training_candidates = (
        average_combined * total_source1
    )

    print("\nTRAINING-SCALE ESTIMATE")
    print("-" * 60)

    print(
        f"Average candidates / Source 1 : "
        f"{average_combined:,.2f}"
    )

    print(
        f"Estimated candidates across "
        f"{total_source1:,} Source 1 records: "
        f"{estimated_training_candidates:,.0f}"
    )

    print("\n")
    print("=" * 70)
    print("Measurement complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()

