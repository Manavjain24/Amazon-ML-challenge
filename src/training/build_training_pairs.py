import csv
import os
import random

# ============================================================
# CONFIG
# ============================================================

GROUND_TRUTH_PATH = "./data/dataset/train/train_ground_truth.tsv"
SOURCE2_PATH = "./data/dataset/train/train_source2.tsv"
SOURCE3_PATH = "./data/dataset/train/train_source3.tsv"

OUTPUT_PATH = "./data/training_pairs.tsv"

MAX_POSITIVE_PAIRS = 30_000
TOTAL_NEGATIVE_PAIRS = 90_000

# Split negatives between Source 2 and Source 3.
SOURCE2_NEGATIVES = TOTAL_NEGATIVE_PAIRS // 2
SOURCE3_NEGATIVES = (
    TOTAL_NEGATIVE_PAIRS - SOURCE2_NEGATIVES
)

RANDOM_SEED = 42


# ============================================================
# RESERVOIR SAMPLING
# ============================================================

def reservoir_add(
    reservoir,
    item,
    seen_count,
    target_size,
    rng,
):
    """
    Reservoir sampling.

    Keeps a uniformly sampled subset of target_size items
    without loading the entire input file into memory.

    Returns updated seen_count.
    """

    seen_count += 1

    if len(reservoir) < target_size:
        reservoir.append(item)

    else:
        replacement_index = rng.randrange(
            seen_count
        )

        if replacement_index < target_size:
            reservoir[replacement_index] = item

    return seen_count


# ============================================================
# SAMPLE POSITIVE RELATIONSHIPS
# ============================================================

def sample_positive_pairs():

    print("=" * 70)
    print("STEP 1: SAMPLING POSITIVE RELATIONSHIPS")
    print("=" * 70)

    rng = random.Random(
        RANDOM_SEED
    )

    reservoir = []
    seen_relationships = 0
    gt_rows = 0
    rows_with_matches = 0

    with open(
        GROUND_TRUTH_PATH,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        for row in reader:

            gt_rows += 1

            source1_id = row[
                "source1_entity_id"
            ]

            matched_ids = row[
                "matched_entity_ids"
            ]

            if not matched_ids:
                continue

            rows_with_matches += 1

            for target_id in matched_ids.split(","):

                target_id = target_id.strip()

                if not target_id:
                    continue

                seen_relationships = reservoir_add(
                    reservoir,
                    (
                        source1_id,
                        target_id,
                    ),
                    seen_relationships,
                    MAX_POSITIVE_PAIRS,
                    rng,
                )

    print(
        f"Ground truth rows read: "
        f"{gt_rows:,}"
    )

    print(
        f"Rows with matches: "
        f"{rows_with_matches:,}"
    )

    print(
        f"Total relationships encountered: "
        f"{seen_relationships:,}"
    )

    print(
        f"Positive pairs selected: "
        f"{len(reservoir):,}"
    )

    return reservoir


# ============================================================
# BUILD POSITIVE LOOKUPS
# ============================================================

def build_positive_lookup(
    positive_pairs,
):
    """
    Build only small lookups for the selected 30k positives.

    This is safe for an 8 GB machine.
    """

    positives_by_source1 = {}
    positive_target_ids = set()

    for source1_id, target_id in positive_pairs:

        if source1_id not in positives_by_source1:
            positives_by_source1[
                source1_id
            ] = set()

        positives_by_source1[
            source1_id
        ].add(target_id)

        positive_target_ids.add(
            target_id
        )

    return (
        positives_by_source1,
        positive_target_ids,
    )


# ============================================================
# SAMPLE NEGATIVES FROM A SOURCE FILE
# ============================================================

def sample_negative_records(
    path,
    target_size,
    positive_target_ids,
    rng,
    source_name,
):
    """
    Stream through one source file and reservoir-sample
    records that are NOT known positive targets.

    Only entity_id is stored.

    No pandas.
    No DataFrame.
    No blocking index.
    """

    print()
    print(
        f"Sampling {target_size:,} negatives "
        f"from {source_name}..."
    )

    reservoir = []
    seen_eligible = 0
    total_rows = 0

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

        for row in reader:

            total_rows += 1

            entity_id = row[
                "entity_id"
            ]

            # Never use a known positive target
            # as a negative.
            if entity_id in positive_target_ids:
                continue

            seen_eligible = reservoir_add(
                reservoir,
                entity_id,
                seen_eligible,
                target_size,
                rng,
            )

            # Progress every 1M records.
            if total_rows % 1_000_000 == 0:

                print(
                    f"{source_name}: "
                    f"{total_rows:,} rows scanned | "
                    f"eligible={seen_eligible:,}",
                    flush=True,
                )

    print(
        f"{source_name}: "
        f"{total_rows:,} rows scanned"
    )

    print(
        f"{source_name}: "
        f"{len(reservoir):,} negatives selected"
    )

    return reservoir


# ============================================================
# WRITE TRAINING PAIRS
# ============================================================

def write_training_pairs(
    positive_pairs,
    negative_source2,
    negative_source3,
):

    print()
    print("=" * 70)
    print("STEP 3: WRITING TRAINING PAIRS")
    print("=" * 70)

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True,
    )

    positive_count = 0
    negative_count = 0

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.writer(
            f,
            delimiter="\t",
        )

        writer.writerow(
            [
                "source1_entity_id",
                "matched_entity_id",
                "label",
            ]
        )

        # ----------------------------------------------------
        # Positives
        # ----------------------------------------------------

        for source1_id, target_id in positive_pairs:

            writer.writerow(
                [
                    source1_id,
                    target_id,
                    1,
                ]
            )

            positive_count += 1

        # ----------------------------------------------------
        # Negatives
        #
        # Randomly assign selected negatives to selected
        # Source 1 entities.
        # ----------------------------------------------------

        rng = random.Random(
            RANDOM_SEED + 1
        )

        selected_source1_ids = list(
            {
                source1_id
                for source1_id, _ in positive_pairs
            }
        )

        # -----------------------------------------------
        # Source 2 negatives
        # -----------------------------------------------

        for target_id in negative_source2:

            source1_id = rng.choice(
                selected_source1_ids
            )

            writer.writerow(
                [
                    source1_id,
                    target_id,
                    0,
                ]
            )

            negative_count += 1

        # -----------------------------------------------
        # Source 3 negatives
        # -----------------------------------------------

        for target_id in negative_source3:

            source1_id = rng.choice(
                selected_source1_ids
            )

            writer.writerow(
                [
                    source1_id,
                    target_id,
                    0,
                ]
            )

            negative_count += 1

    print(
        f"Positive pairs written: "
        f"{positive_count:,}"
    )

    print(
        f"Negative pairs written: "
        f"{negative_count:,}"
    )

    print(
        f"Total pairs written: "
        f"{positive_count + negative_count:,}"
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("       MEMORY-SAFE TRAINING PAIR GENERATION")
    print("=" * 70)

    rng = random.Random(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # 1. Select positives
    # --------------------------------------------------------

    positive_pairs = (
        sample_positive_pairs()
    )

    # --------------------------------------------------------
    # 2. Build tiny positive lookup
    # --------------------------------------------------------

    (
        positives_by_source1,
        positive_target_ids,
    ) = build_positive_lookup(
        positive_pairs
    )

    print()
    print(
        f"Selected Source 1 entities: "
        f"{len(positives_by_source1):,}"
    )

    print(
        f"Unique positive target IDs: "
        f"{len(positive_target_ids):,}"
    )

    # --------------------------------------------------------
    # 3. Stream Source 2
    # --------------------------------------------------------

    negative_source2 = (
        sample_negative_records(
            SOURCE2_PATH,
            SOURCE2_NEGATIVES,
            positive_target_ids,
            rng,
            "Source 2",
        )
    )

    # --------------------------------------------------------
    # 4. Stream Source 3
    # --------------------------------------------------------

    negative_source3 = (
        sample_negative_records(
            SOURCE3_PATH,
            SOURCE3_NEGATIVES,
            positive_target_ids,
            rng,
            "Source 3",
        )
    )

    # --------------------------------------------------------
    # 5. Write
    # --------------------------------------------------------

    write_training_pairs(
        positive_pairs,
        negative_source2,
        negative_source3,
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()