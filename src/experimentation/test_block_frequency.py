import csv
import time
from collections import defaultdict

from src.blocking.candidate_generation import create_block_keys


# ============================================================
# CONFIG
# ============================================================

GROUND_TRUTH = "./data/dataset/train/train_ground_truth.tsv"
SOURCE1 = "./data/dataset/train/train_source1.tsv"
SOURCE2 = "./data/dataset/train/train_source2.tsv"
SOURCE3 = "./data/dataset/train/train_source3.tsv"

# Same evaluation size we have already been using.
MAX_GT_ROWS = 10_000

# Frequency thresholds to test.
# A block containing MORE records than the threshold
# will be ignored.
THRESHOLDS = [
    25,
    50,
    100,
    250,
    500,
    1000,
    2500,
    5000,
    10000,
]


# ============================================================
# LOAD GROUND TRUTH SAMPLE
# ============================================================

def load_ground_truth_sample():
    print("=" * 70)
    print("LOADING GROUND TRUTH SAMPLE")
    print("=" * 70)

    gt = []

    required_s1 = set()
    required_s2 = set()
    required_s3 = set()

    with open(
        GROUND_TRUTH,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        for row in reader:

            if not row["matched_entity_ids"]:
                continue

            gt.append(row)

            required_s1.add(
                row["source1_entity_id"]
            )

            for entity_id in row["matched_entity_ids"].split(","):

                entity_id = entity_id.strip()

                if entity_id.startswith("S2-"):
                    required_s2.add(entity_id)

                elif entity_id.startswith("S3-"):
                    required_s3.add(entity_id)

            if len(gt) >= MAX_GT_ROWS:
                break

    print(f"Ground truth rows: {len(gt):,}")
    print(f"Required Source 1: {len(required_s1):,}")
    print(f"Required Source 2: {len(required_s2):,}")
    print(f"Required Source 3: {len(required_s3):,}")

    return (
        gt,
        required_s1,
        required_s2,
        required_s3,
    )


# ============================================================
# LOAD REQUIRED SOURCE RECORDS
# ============================================================

def load_required_records(path, required_ids, source_name):
    print()
    print(f"Loading required {source_name} records...")

    records = {}

    with open(
        path,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        scanned = 0

        for row in reader:

            scanned += 1

            entity_id = row["entity_id"]

            if entity_id in required_ids:

                records[entity_id] = row

            if scanned % 1_000_000 == 0:

                print(
                    f"{source_name}: "
                    f"{scanned:,} rows scanned | "
                    f"found={len(records):,}",
                    flush=True,
                )

            if len(records) == len(required_ids):
                break

    print(
        f"{source_name} records loaded: "
        f"{len(records):,}"
    )

    return records


# ============================================================
# BUILD SOURCE 1 BLOCK INDEX
# ============================================================

def build_source1_index():
    print()
    print("=" * 70)
    print("BUILDING SOURCE 1 BLOCK INDEX")
    print("=" * 70)

    block_index = defaultdict(list)

    total_rows = 0
    total_keys = 0

    start = time.time()

    with open(
        SOURCE1,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        for row in reader:

            entity_id = row["entity_id"]

            keys = create_block_keys(
                row["country"],
                row["business_name"],
                row["business_address"],
            )

            for key in keys:

                block_index[key].append(
                    entity_id
                )

                total_keys += 1

            total_rows += 1

            if total_rows % 250_000 == 0:

                elapsed = time.time() - start

                rate = (
                    total_rows / elapsed
                    if elapsed > 0
                    else 0
                )

                print(
                    f"Source 1: "
                    f"{total_rows:,} rows | "
                    f"{total_keys:,} keys | "
                    f"{rate:,.0f} rows/sec",
                    flush=True,
                )

    print()
    print(
        f"Source 1 rows indexed: {total_rows:,}"
    )

    print(
        f"Unique blocking keys: "
        f"{len(block_index):,}"
    )

    print(
        f"Total key assignments: "
        f"{total_keys:,}"
    )

    return dict(block_index)


# ============================================================
# BLOCK STATISTICS
# ============================================================

def build_block_sizes(block_index):

    return {
        key: len(entity_ids)
        for key, entity_ids
        in block_index.items()
    }


# ============================================================
# CANDIDATE RETRIEVAL
# ============================================================

def get_candidates(
    record,
    block_index,
    block_sizes,
    max_block_size,
):

    keys = create_block_keys(
        record["country"],
        record["business_name"],
        record["business_address"],
    )

    candidates = set()

    used_keys = 0

    for key in keys:

        size = block_sizes.get(key, 0)

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # We do NOT arbitrarily take the first N records.
        #
        # We either use the complete block or discard it.
        # ----------------------------------------------------

        if size == 0:
            continue

        if size > max_block_size:
            continue

        used_keys += 1

        candidates.update(
            block_index[key]
        )

    return candidates, used_keys


# ============================================================
# EVALUATE ONE THRESHOLD
# ============================================================

def evaluate_threshold(
    threshold,
    gt_rows,
    source2_records,
    source3_records,
    block_index,
    block_sizes,
):

    found = 0
    missed = 0

    total_gt_relationships = 0

    total_candidates = 0
    total_source_records = 0

    used_key_count = 0

    max_candidates = 0

    for row in gt_rows:

        s2_ids = [
            entity_id.strip()
            for entity_id
            in row["matched_entity_ids"].split(",")
            if entity_id.startswith("S2-")
        ]

        s3_ids = [
            entity_id.strip()
            for entity_id
            in row["matched_entity_ids"].split(",")
            if entity_id.startswith("S3-")
        ]

        total_gt_relationships += (
            len(s2_ids) + len(s3_ids)
        )

        # ----------------------------------------------------
        # Test Source 2 relationships.
        # ----------------------------------------------------

        for entity_id in s2_ids:

            record = source2_records.get(
                entity_id
            )

            if record is None:
                continue

            candidates, used_keys = get_candidates(
                record,
                block_index,
                block_sizes,
                threshold,
            )

            total_candidates += len(candidates)
            max_candidates = max(
                max_candidates,
                len(candidates),
            )

            used_key_count += used_keys
            total_source_records += 1

            if row["source1_entity_id"] in candidates:
                found += 1
            else:
                missed += 1

        # ----------------------------------------------------
        # Test Source 3 relationships.
        # ----------------------------------------------------

        for entity_id in s3_ids:

            record = source3_records.get(
                entity_id
            )

            if record is None:
                continue

            candidates, used_keys = get_candidates(
                record,
                block_index,
                block_sizes,
                threshold,
            )

            total_candidates += len(candidates)
            max_candidates = max(
                max_candidates,
                len(candidates),
            )

            used_key_count += used_keys
            total_source_records += 1

            if row["source1_entity_id"] in candidates:
                found += 1
            else:
                missed += 1

    recall = (
        found / total_gt_relationships
        if total_gt_relationships
        else 0
    )

    average_candidates = (
        total_candidates / total_source_records
        if total_source_records
        else 0
    )

    average_keys = (
        used_key_count / total_source_records
        if total_source_records
        else 0
    )

    return {
        "threshold": threshold,
        "relationships": total_gt_relationships,
        "found": found,
        "missed": missed,
        "recall": recall,
        "avg_candidates": average_candidates,
        "max_candidates": max_candidates,
        "avg_used_keys": average_keys,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("       BLOCK FREQUENCY THRESHOLD EXPERIMENT")
    print("=" * 70)

    # --------------------------------------------------------
    # Ground truth.
    # --------------------------------------------------------

    (
        gt_rows,
        required_s1,
        required_s2,
        required_s3,
    ) = load_ground_truth_sample()

    # --------------------------------------------------------
    # Source 2 / Source 3 records.
    # --------------------------------------------------------

    source2_records = load_required_records(
        SOURCE2,
        required_s2,
        "Source 2",
    )

    source3_records = load_required_records(
        SOURCE3,
        required_s3,
        "Source 3",
    )

    # --------------------------------------------------------
    # Source 1 index.
    # --------------------------------------------------------

    block_index = build_source1_index()

    block_sizes = build_block_sizes(
        block_index
    )

    # --------------------------------------------------------
    # Block distribution.
    # --------------------------------------------------------

    sizes = list(
        block_sizes.values()
    )

    sizes.sort()

    print()
    print("=" * 70)
    print("BLOCK SIZE DISTRIBUTION")
    print("=" * 70)

    if sizes:

        def percentile(values, p):
            index = int(
                (len(values) - 1) * p
            )
            return values[index]

        print(
            f"Min:    {sizes[0]:,}"
        )

        print(
            f"Median: "
            f"{percentile(sizes, 0.50):,}"
        )

        print(
            f"P90:    "
            f"{percentile(sizes, 0.90):,}"
        )

        print(
            f"P95:    "
            f"{percentile(sizes, 0.95):,}"
        )

        print(
            f"P99:    "
            f"{percentile(sizes, 0.99):,}"
        )

        print(
            f"Max:    {sizes[-1]:,}"
        )

    # --------------------------------------------------------
    # Threshold sweep.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("THRESHOLD SWEEP")
    print("=" * 70)

    print(
        f"{'THRESHOLD':>10} "
        f"{'RECALL':>10} "
        f"{'FOUND':>10} "
        f"{'MISSED':>10} "
        f"{'AVG CAND':>12} "
        f"{'MAX CAND':>12} "
        f"{'AVG KEYS':>10}"
    )

    print("-" * 82)

    for threshold in THRESHOLDS:

        start = time.time()

        result = evaluate_threshold(
            threshold,
            gt_rows,
            source2_records,
            source3_records,
            block_index,
            block_sizes,
        )

        elapsed = time.time() - start

        print(
            f"{result['threshold']:>10} "
            f"{result['recall'] * 100:>9.3f}% "
            f"{result['found']:>10,} "
            f"{result['missed']:>10,} "
            f"{result['avg_candidates']:>12.1f} "
            f"{result['max_candidates']:>12,} "
            f"{result['avg_used_keys']:>10.2f} "
            f"({elapsed:.1f}s)",
            flush=True,
        )

    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()