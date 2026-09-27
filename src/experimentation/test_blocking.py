import pandas as pd

from src.blocking.candidate_generation import (
    build_block_index,
    get_candidates
)


SOURCE1_FILE = "data/dataset/train/train_source1.tsv"
SOURCE2_FILE = "data/dataset/train/train_source2.tsv"
SOURCE3_FILE = "data/dataset/train/train_source3.tsv"

SAMPLE_SIZE = 100


# --------------------------------------------------
# Load Source 1
# --------------------------------------------------

print("Loading Source 1...")

source1_df = pd.read_csv(
    SOURCE1_FILE,
    sep="\t"
)

print("Source 1 records:", len(source1_df))


# --------------------------------------------------
# Build Source 1 block index
# --------------------------------------------------

print("\nBuilding Source 1 block index...")

block_index = build_block_index(source1_df)

print("Number of blocks:", len(block_index))


# --------------------------------------------------
# Block statistics
# --------------------------------------------------

block_sizes = [
    len(indices)
    for indices in block_index.values()
]

block_sizes_series = pd.Series(block_sizes)

print("\nBlock size statistics:")
print("Minimum:", block_sizes_series.min())
print("Maximum:", block_sizes_series.max())
print("Average:", block_sizes_series.mean())
print("Median:", block_sizes_series.median())
print(
    "Blocks with exactly 1 record:",
    (block_sizes_series == 1).sum()
)
print(
    "Blocks with more than 10 records:",
    (block_sizes_series > 10).sum()
)


# --------------------------------------------------
# Load small samples of Source 2 and Source 3
# --------------------------------------------------

print("\nLoading Source 2 sample...")

source2_df = pd.read_csv(
    SOURCE2_FILE,
    sep="\t",
    nrows=SAMPLE_SIZE
)

print("Source 2 sample records:", len(source2_df))


print("\nLoading Source 3 sample...")

source3_df = pd.read_csv(
    SOURCE3_FILE,
    sep="\t",
    nrows=SAMPLE_SIZE
)

print("Source 3 sample records:", len(source3_df))


# --------------------------------------------------
# Test Source 2
# --------------------------------------------------

print("\n\n================ SOURCE 2 TEST ================")

source2_record = source2_df.iloc[0]

source2_candidates = get_candidates(
    source2_record,
    block_index
)

print("Source 2 record:")
print("Entity ID:", source2_record["entity_id"])
print("Business name:", source2_record["business_name"])
print("Country:", source2_record["country"])

print("\nCandidate Source 1 indices:")
print(source2_candidates)

print("\nNumber of candidates:", len(source2_candidates))

print("\nCandidate Source 1 records:")

for idx in source2_candidates[:20]:

    print(
        idx,
        source1_df.iloc[idx]["entity_id"],
        source1_df.iloc[idx]["business_name"],
        source1_df.iloc[idx]["country"]
    )


# --------------------------------------------------
# Test Source 3
# --------------------------------------------------

print("\n\n================ SOURCE 3 TEST ================")

source3_record = source3_df.iloc[0]

source3_candidates = get_candidates(
    source3_record,
    block_index
)

print("Source 3 record:")
print("Entity ID:", source3_record["entity_id"])
print("Business name:", source3_record["business_name"])
print("Country:", source3_record["country"])

print("\nCandidate Source 1 indices:")
print(source3_candidates)

print("\nNumber of candidates:", len(source3_candidates))

print("\nCandidate Source 1 records:")

for idx in source3_candidates[:20]:

    print(
        idx,
        source1_df.iloc[idx]["entity_id"],
        source1_df.iloc[idx]["business_name"],
        source1_df.iloc[idx]["country"]
    )


# --------------------------------------------------
# Candidate statistics over samples
# --------------------------------------------------

print("\n\n================ SAMPLE CANDIDATE STATISTICS ================")


def calculate_candidate_stats(source_df, source_name):

    candidate_counts = []

    for _, record in source_df.iterrows():

        candidates = get_candidates(
            record,
            block_index
        )

        candidate_counts.append(len(candidates))

    counts = pd.Series(candidate_counts)

    print(f"\n{source_name}:")
    print("Records tested:", len(source_df))
    print("Minimum candidates:", counts.min())
    print("Maximum candidates:", counts.max())
    print("Average candidates:", counts.mean())
    print("Median candidates:", counts.median())
    print("Records with 0 candidates:", (counts == 0).sum())
    print(
        "Records with >10 candidates:",
        (counts > 10).sum()
    )


calculate_candidate_stats(
    source2_df,
    "Source 2"
)

calculate_candidate_stats(
    source3_df,
    "Source 3"
)