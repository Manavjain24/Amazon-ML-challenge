import csv
import os
import sys

from src.features.pair_features import extract_pair_features


# ============================================================
# CONFIGURATION
# ============================================================

TRAINING_PAIRS_PATH = "./data/training_pairs.tsv"

SOURCE1_PATH = "./data/dataset/train/train_source1.tsv"
SOURCE2_PATH = "./data/dataset/train/train_source2.tsv"
SOURCE3_PATH = "./data/dataset/train/train_source3.tsv"

OUTPUT_PATH = "./data/training_pairs.tsv"

# ============================================================
# LOAD TRAINING PAIR IDs
# ============================================================

def load_training_pairs(path):
    """
    Load only the IDs required for feature extraction.

    We do NOT load the source datasets here.
    """

    pairs = []

    source1_ids = set()
    source2_ids = set()
    source3_ids = set()

    print("Loading training pairs...")

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

            source1_id = row["source1_entity_id"]
            target_id = row["matched_entity_id"]
            label = int(row["label"])

            pairs.append(
                (
                    source1_id,
                    target_id,
                    label,
                )
            )

            source1_ids.add(source1_id)

            if target_id.startswith("S2-"):
                source2_ids.add(target_id)

            elif target_id.startswith("S3-"):
                source3_ids.add(target_id)

            else:
                raise ValueError(
                    f"Unknown target entity ID: {target_id}"
                )

    print(
        f"Training pairs: {len(pairs):,}"
    )

    print(
        f"Required Source 1 records: "
        f"{len(source1_ids):,}"
    )

    print(
        f"Required Source 2 records: "
        f"{len(source2_ids):,}"
    )

    print(
        f"Required Source 3 records: "
        f"{len(source3_ids):,}"
    )

    return (
        pairs,
        source1_ids,
        source2_ids,
        source3_ids,
    )


# ============================================================
# STREAM ONE SOURCE FILE
# ============================================================

def load_required_records(
    path,
    required_ids,
    source_name,
):
    """
    Stream a source TSV and keep only records required
    by the training pairs.

    This is deliberately NOT pandas-based.

    Memory usage is proportional to the selected records,
    not the size of the source file.
    """

    records = {}

    total_rows = 0

    print()
    print(
        f"Loading required {source_name} records..."
    )

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

            entity_id = row["entity_id"]

            if entity_id in required_ids:

                records[entity_id] = row

            if total_rows % 1_000_000 == 0:

                print(
                    f"{source_name}: "
                    f"{total_rows:,} rows scanned | "
                    f"found={len(records):,}",
                    flush=True,
                )

            # Once every required record has been found,
            # there is no reason to continue scanning.
            if len(records) == len(required_ids):
                break

    print(
        f"{source_name}: "
        f"{total_rows:,} rows scanned"
    )

    print(
        f"{source_name} records loaded: "
        f"{len(records):,}"
    )

    missing = len(required_ids) - len(records)

    if missing:
        print(
            f"WARNING: {missing:,} required "
            f"{source_name} records were not found."
        )

    return records


# ============================================================
# WRITE FEATURES
# ============================================================

def write_features(
    pairs,
    source1_records,
    source2_records,
    source3_records,
):
    """
    Generate pair features and write them incrementally.

    We keep the output on disk instead of accumulating a
    120k-row feature list in memory.
    """

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True,
    )

    print()
    print("Generating pair features...")

    feature_names = None

    processed = 0
    skipped = 0

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

        for source1_id, target_id, label in pairs:

            source1_record = source1_records.get(
                source1_id
            )

            if source1_record is None:

                skipped += 1
                continue

            # ------------------------------------------------
            # Determine target source.
            # ------------------------------------------------

            if target_id.startswith("S2-"):

                target_record = source2_records.get(
                    target_id
                )

            elif target_id.startswith("S3-"):

                target_record = source3_records.get(
                    target_id
                )

            else:

                skipped += 1
                continue

            if target_record is None:

                skipped += 1
                continue

            # ------------------------------------------------
            # Extract features.
            # ------------------------------------------------

            features = extract_pair_features(
                source1_record,
                target_record,
            )

            # ------------------------------------------------
            # Write header once.
            # ------------------------------------------------

            if feature_names is None:

                feature_names = list(
                    features.keys()
                )

                writer.writerow(
                    [
                        "source1_entity_id",
                        "matched_entity_id",
                        "label",
                    ]
                    + feature_names
                )

            # ------------------------------------------------
            # Write row.
            # ------------------------------------------------

            writer.writerow(
                [
                    source1_id,
                    target_id,
                    label,
                ]
                + [
                    features[name]
                    for name in feature_names
                ]
            )

            processed += 1

            if processed % 5_000 == 0:

                print(
                    f"Processed "
                    f"{processed:,}/"
                    f"{len(pairs):,}",
                    flush=True,
                )

    print()
    print(
        f"Feature rows written: "
        f"{processed:,}"
    )

    print(
        f"Pairs skipped: "
        f"{skipped:,}"
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("          MEMORY-SAFE TRAINING FEATURE GENERATION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Read pair IDs
    # --------------------------------------------------------

    (
        pairs,
        source1_ids,
        source2_ids,
        source3_ids,
    ) = load_training_pairs(
        TRAINING_PAIRS_PATH
    )

    # --------------------------------------------------------
    # 2. Stream Source 1
    # --------------------------------------------------------

    source1_records = load_required_records(
        SOURCE1_PATH,
        source1_ids,
        "Source 1",
    )

    # --------------------------------------------------------
    # 3. Stream Source 2
    # --------------------------------------------------------

    source2_records = load_required_records(
        SOURCE2_PATH,
        source2_ids,
        "Source 2",
    )

    # --------------------------------------------------------
    # 4. Stream Source 3
    # --------------------------------------------------------

    source3_records = load_required_records(
        SOURCE3_PATH,
        source3_ids,
        "Source 3",
    )

    # --------------------------------------------------------
    # 5. Generate features
    # --------------------------------------------------------

    write_features(
        pairs,
        source1_records,
        source2_records,
        source3_records,
    )

    print()
    print("=" * 70)
    print("FEATURE GENERATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()