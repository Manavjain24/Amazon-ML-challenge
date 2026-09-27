import csv
import os
import sqlite3
import sys
import time

import joblib
import pandas as pd

from src.blocking.candidate_generation import create_block_keys
from src.features.pair_features import extract_pair_features


# ============================================================
# CONFIGURATION
# ============================================================

TEST_SOURCE1 = "./data/dataset/test/test_source1.tsv"
TEST_SOURCE2 = "./data/dataset/test/test_source2.tsv"
TEST_SOURCE3 = "./data/dataset/test/test_source3.tsv"

MODEL_PATH = "./data/entity_resolution_model.joblib"

OUTPUT_DIR = "./output"

MATCHING_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "matching_results.tsv",
)

CANDIDATE_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "candidate_pairs.tsv",
)

SQLITE_DB = "./data/test_blocking.db"


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

MODEL_THRESHOLD = 0.80


# ------------------------------------------------------------
# BLOCKING
# ------------------------------------------------------------

# Prevent one pathological block from exploding inference.
#
# We measured block sizes up to 371,738, so allowing 5,000
# records from a single block is too expensive on an 8 GB machine.
MAX_CANDIDATES_PER_KEY = 500


# ------------------------------------------------------------
# SECOND-STAGE SCORING
# ------------------------------------------------------------

# After blocking, candidates are ranked by the number of
# independent blocking keys that produced them.
#
# Only the strongest candidates are sent through the expensive
# feature extraction + ML scoring stage.
MAX_EXPENSIVE_CANDIDATES = 300


# ------------------------------------------------------------
# SQLITE
# ------------------------------------------------------------

COMMIT_EVERY = 50_000


# ============================================================
# SQLITE SETUP
# ============================================================

def create_database():
    """
    Create a disk-backed blocking database.

    This avoids keeping millions of Python dictionary/list
    objects in RAM.
    """

    if os.path.exists(SQLITE_DB):

        print(
            f"Removing existing SQLite database: "
            f"{SQLITE_DB}"
        )

        os.remove(SQLITE_DB)

    conn = sqlite3.connect(
        SQLITE_DB
    )

    conn.execute(
        "PRAGMA journal_mode=WAL"
    )

    conn.execute(
        "PRAGMA synchronous=OFF"
    )

    conn.execute(
        "PRAGMA temp_store=FILE"
    )

    conn.execute(
        "PRAGMA cache_size=-100000"
    )

    conn.execute(
        """
        CREATE TABLE records (
            entity_id TEXT PRIMARY KEY,
            business_name TEXT,
            business_address TEXT,
            country TEXT
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE block_index (
            block_key TEXT,
            entity_id TEXT
        )
        """
    )

    return conn


# ============================================================
# BUILD SOURCE INDEX
# ============================================================

def index_source(
    conn,
    path,
    source_name,
):
    """
    Stream a source TSV and insert:

        1. record itself
        2. all blocking keys

    Nothing from the entire source is loaded into memory.
    """

    print()
    print("=" * 70)
    print(
        f"INDEXING {source_name}"
    )
    print("=" * 70)

    insert_record = conn.execute

    total_rows = 0
    total_keys = 0

    start = time.time()

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

            entity_id = row["entity_id"]

            business_name = row[
                "business_name"
            ]

            business_address = row[
                "business_address"
            ]

            country = row[
                "country"
            ]

            # ------------------------------------------------
            # Store record.
            # ------------------------------------------------

            insert_record(
                """
                INSERT INTO records
                VALUES (?, ?, ?, ?)
                """,
                (
                    entity_id,
                    business_name,
                    business_address,
                    country,
                ),
            )

            # ------------------------------------------------
            # Generate blocking keys.
            # ------------------------------------------------

            keys = create_block_keys(
                country,
                business_name,
                business_address,
            )

            for key in keys:

                insert_record(
                    """
                    INSERT INTO block_index
                    VALUES (?, ?)
                    """,
                    (
                        key,
                        entity_id,
                    ),
                )

                total_keys += 1

            total_rows += 1

            if total_rows % COMMIT_EVERY == 0:

                conn.commit()

            if total_rows % 250_000 == 0:

                elapsed = (
                    time.time() - start
                )

                rate = (
                    total_rows / elapsed
                    if elapsed > 0
                    else 0
                )

                print(
                    f"{source_name}: "
                    f"{total_rows:,} rows | "
                    f"{total_keys:,} keys | "
                    f"{rate:,.0f} rows/sec",
                    flush=True,
                )

    conn.commit()

    print()

    print(
        f"{source_name} indexing complete."
    )

    print(
        f"Rows indexed: {total_rows:,}"
    )

    print(
        f"Blocking keys: {total_keys:,}"
    )

    print(
        f"Time: {time.time() - start:.1f}s"
    )


# ============================================================
# FINALIZE DATABASE
# ============================================================

def finalize_database(conn):
    """
    Create the lookup index after all inserts are complete.

    Creating this index incrementally would make insertion much
    slower, so we deliberately create it once at the end.
    """

    print()
    print(
        "Creating SQLite blocking index..."
    )

    start = time.time()

    conn.execute(
        """
        CREATE INDEX idx_block_key
        ON block_index(block_key)
        """
    )

    conn.commit()

    print(
        f"SQLite index created in "
        f"{time.time() - start:.1f}s"
    )


# ============================================================
# FETCH CANDIDATES
# ============================================================

def get_candidate_ids(
    conn,
    source_record,
):
    """
    Retrieve candidates from blocking.

    Returns:

        scoring_candidate_ids
            Small shortlist used for expensive feature extraction
            and model scoring.

        all_candidate_ids
            All candidates found by blocking, used for the
            candidate-pairs output.

    Candidate ranking is based on the number of blocking keys
    that independently matched the candidate.
    """

    keys = create_block_keys(
        source_record["country"],
        source_record["business_name"],
        source_record["business_address"],
    )

    if not keys:
        return [], []

    cursor = conn.cursor()

    candidate_hits = {}

    # ========================================================
    # COLLECT BLOCKING CANDIDATES
    # ========================================================

    for key in keys:

        rows = cursor.execute(
            """
            SELECT entity_id
            FROM block_index
            WHERE block_key = ?
            LIMIT ?
            """,
            (
                key,
                MAX_CANDIDATES_PER_KEY,
            ),
        )

        for row in rows:

            entity_id = row[0]

            candidate_hits[entity_id] = (
                candidate_hits.get(entity_id, 0) + 1
            )

    if not candidate_hits:
        return [], []

    # ========================================================
    # ALL CANDIDATES
    # ========================================================

    all_candidate_ids = list(
        candidate_hits.keys()
    )

    # ========================================================
    # RANK CANDIDATES
    # ========================================================
    #
    # More independent blocking-key hits = stronger candidate.
    #
    # We do NOT run expensive string similarity here.
    # ========================================================

    ranked_candidates = sorted(
        candidate_hits.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )

    # ========================================================
    # EXPENSIVE SCORING SHORTLIST
    # ========================================================

    MAX_EXPENSIVE_CANDIDATES = 30

    scoring_candidate_ids = [
        entity_id
        for entity_id, hit_count
        in ranked_candidates[
            :MAX_EXPENSIVE_CANDIDATES
        ]
    ]

    return (
        scoring_candidate_ids,
        all_candidate_ids,
    )

# ============================================================
# FETCH RECORDS
# ============================================================

def fetch_records(
    conn,
    entity_ids,
):
    """
    Fetch candidate records from SQLite.

    Uses batched queries rather than querying once per candidate.
    """

    if not entity_ids:

        return {}

    result = {}

    cursor = conn.cursor()

    ids = list(entity_ids)

    batch_size = 500

    for start in range(
        0,
        len(ids),
        batch_size,
    ):

        batch = ids[
            start:start + batch_size
        ]

        placeholders = ",".join(
            "?" for _ in batch
        )

        query = f"""
            SELECT
                entity_id,
                business_name,
                business_address,
                country
            FROM records
            WHERE entity_id IN ({placeholders})
        """

        rows = cursor.execute(
            query,
            batch,
        )

        for row in rows:

            result[row[0]] = {
                "entity_id": row[0],
                "business_name": row[1],
                "business_address": row[2],
                "country": row[3],
            }

    return result


# ============================================================
# SCORE CANDIDATES
# ============================================================

def _cheap_candidate_score(
    source1_record,
    candidate_record,
):
    """
    Cheap deterministic score used ONLY to rank candidates
    before expensive feature extraction.

    This is not the final match score.

    Higher score = stronger basic agreement.
    """

    score = 0

    name1 = (
        str(source1_record["business_name"])
        .strip()
        .lower()
    )

    name2 = (
        str(candidate_record["business_name"])
        .strip()
        .lower()
    )

    address1 = (
        str(source1_record["business_address"])
        .strip()
        .lower()
    )

    address2 = (
        str(candidate_record["business_address"])
        .strip()
        .lower()
    )

    country1 = (
        str(source1_record["country"])
        .strip()
        .lower()
    )

    country2 = (
        str(candidate_record["country"])
        .strip()
        .lower()
    )

    # --------------------------------------------------------
    # Country agreement
    # --------------------------------------------------------

    if country1 and country1 == country2:
        score += 5

    # --------------------------------------------------------
    # Exact name agreement
    # --------------------------------------------------------

    if name1 and name1 == name2:
        score += 100

    # --------------------------------------------------------
    # Exact address agreement
    # --------------------------------------------------------

    if address1 and address1 == address2:
        score += 80

    # --------------------------------------------------------
    # Name token overlap using cheap Python sets.
    # --------------------------------------------------------

    name_tokens1 = set(
        name1.split()
    )

    name_tokens2 = set(
        name2.split()
    )

    if name_tokens1 and name_tokens2:

        overlap = len(
            name_tokens1 & name_tokens2
        )

        score += min(
            overlap * 10,
            50,
        )

    # --------------------------------------------------------
    # Address token overlap.
    # --------------------------------------------------------

    address_tokens1 = set(
        address1.split()
    )

    address_tokens2 = set(
        address2.split()
    )

    if address_tokens1 and address_tokens2:

        overlap = len(
            address_tokens1 & address_tokens2
        )

        score += min(
            overlap * 3,
            30,
        )

    # --------------------------------------------------------
    # Prefix agreement.
    #
    # Cheap signal for names that are similar.
    # --------------------------------------------------------

    if (
        len(name1) >= 3
        and len(name2) >= 3
        and name1[:3] == name2[:3]
    ):
        score += 15

    return score


def score_candidates(
    model,
    feature_columns,
    source1_record,
    candidate_records,
):
    """
    Two-stage candidate scoring.

    Stage 1:
        Cheap deterministic ranking.

    Stage 2:
        Expensive feature extraction + trained model
        on only the strongest candidates.

    This is required for practical inference on an
    8 GB RAM / lower-end machine.
    """

    if not candidate_records:
        return []

    # ========================================================
    # STAGE 1
    # CHEAP CANDIDATE RANKING
    # ========================================================

    ranked_candidates = []

    for candidate_id, candidate_record in (
        candidate_records.items()
    ):

        cheap_score = _cheap_candidate_score(
            source1_record,
            candidate_record,
        )

        ranked_candidates.append(
            (
                cheap_score,
                candidate_id,
                candidate_record,
            )
        )

    ranked_candidates.sort(
        key=lambda item: (
            -item[0],
            item[1],
        )
    )

    # ========================================================
    # STAGE 2
    # EXPENSIVE FEATURE EXTRACTION
    # ========================================================

    # Only the strongest candidates reach:
    #
    # normalize
    # tokenize
    # Jaccard
    # Levenshtein
    # transliteration
    # skeleton
    # etc.

    expensive_candidates = ranked_candidates[
        :30
    ]

    candidate_ids = []
    feature_rows = []

    for (
        cheap_score,
        candidate_id,
        candidate_record,
    ) in expensive_candidates:

        features = extract_pair_features(
            source1_record,
            candidate_record,
        )

        feature_vector = {
            column: features[column]
            for column in feature_columns
        }

        candidate_ids.append(
            candidate_id
        )

        feature_rows.append(
            feature_vector
        )

    if not feature_rows:
        return []

    # ========================================================
    # BATCH MODEL PREDICTION
    # ========================================================

    feature_df = pd.DataFrame(
        feature_rows,
        columns=feature_columns,
    )

    probabilities = model.predict_proba(
        feature_df
    )[:, 1]

    matches = []

    for candidate_id, probability in zip(
        candidate_ids,
        probabilities,
    ):

        if probability >= MODEL_THRESHOLD:

            matches.append(
                (
                    candidate_id,
                    float(probability),
                )
            )

    matches.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    return matches


# ============================================================
# WRITE OUTPUT
# ============================================================

def write_output_row(
    matching_writer,
    candidate_writer,
    source1_id,
    scoring_candidate_ids,
    matched_ids,
):
    """
    Write one row for each Source 1 entity.

    candidate_pairs.tsv contains only the candidates that
    actually reach model inference.
    """

    candidate_list = ",".join(
        sorted(scoring_candidate_ids)
    )

    matched_list = ",".join(
        sorted(matched_ids)
    )

    candidate_writer.writerow(
        [
            source1_id,
            candidate_list,
        ]
    )

    matching_writer.writerow(
        [
            source1_id,
            matched_list,
        ]
    )

# ============================================================
# PROCESS TEST SOURCE 1
# ============================================================

def process_test_source1(
    conn,
    model,
    feature_columns,
):
    """
    Stream Test Source 1 and generate final predictions.

    Only one Source 1 record and its shortlisted candidates
    are processed at a time.
    """

    print()
    print("=" * 70)
    print(
        "GENERATING TEST PREDICTIONS"
    )
    print("=" * 70)

    print()
    print(
        f"Maximum candidates per blocking key: "
        f"{MAX_CANDIDATES_PER_KEY}"
    )

    print(
        f"Maximum expensive candidates per Source 1: "
        f"{MAX_EXPENSIVE_CANDIDATES}"
    )

    print(
        f"Model threshold: "
        f"{MODEL_THRESHOLD}"
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    processed = 0
    total_candidates = 0
    total_scored_candidates = 0
    total_matches = 0
    singleton_predictions = 0

    start = time.time()

    with open(
        TEST_SOURCE1,
        "r",
        encoding="utf-8",
        newline="",
    ) as source1_file, \
        open(
            MATCHING_OUTPUT,
            "w",
            encoding="utf-8",
            newline="",
        ) as matching_file, \
        open(
            CANDIDATE_OUTPUT,
            "w",
            encoding="utf-8",
            newline="",
        ) as candidate_file:

        reader = csv.DictReader(
            source1_file,
            delimiter="\t",
        )

        matching_writer = csv.writer(
            matching_file,
            delimiter="\t",
            lineterminator="\n",
        )

        candidate_writer = csv.writer(
            candidate_file,
            delimiter="\t",
            lineterminator="\n",
        )

        # ----------------------------------------------------
        # Required headers.
        # ----------------------------------------------------

        matching_writer.writerow(
            [
                "source1_entity_id",
                "matched_entity_ids",
            ]
        )

        candidate_writer.writerow(
            [
                "source1_entity_id",
                "candidate_entity_ids",
            ]
        )

        # ----------------------------------------------------
        # Process Source 1.
        # ----------------------------------------------------

        for source1_record in reader:

            source1_id = source1_record[
                "entity_id"
            ]

            # -----------------------------------------------
            # Blocking.
            # -----------------------------------------------

            (
                scoring_candidate_ids,
                all_candidate_ids,
            ) = get_candidate_ids(
                conn,
                source1_record,
            )

            total_candidates += len(
                all_candidate_ids
            )

            total_scored_candidates += len(
                scoring_candidate_ids
            )

            # -----------------------------------------------
            # Fetch only candidates that will actually be
            # scored.
            # -----------------------------------------------

            candidate_records = fetch_records(
                conn,
                scoring_candidate_ids,
            )

            # -----------------------------------------------
            # Model scoring.
            # -----------------------------------------------

            matches = score_candidates(
                model,
                feature_columns,
                source1_record,
                candidate_records,
            )

            matched_ids = [
                candidate_id
                for candidate_id, probability
                in matches
            ]

            # -----------------------------------------------
            # Statistics.
            # -----------------------------------------------

            if not matched_ids:

                singleton_predictions += 1

            total_matches += len(
                matched_ids
            )

            # -----------------------------------------------
            # Write immediately.
            #
            # Candidate output contains the complete
            # blocking candidate set.
            #
            # Matching output contains model-approved
            # matches from the shortlisted candidates.
            # -----------------------------------------------

            write_output_row(
                matching_writer,
                candidate_writer,
                source1_id,
                scoring_candidate_ids,
                matched_ids,
            )

            processed += 1

            if processed % 10_000 == 0:

                elapsed = (
                    time.time() - start
                )

                rate = (
                    processed / elapsed
                    if elapsed > 0
                    else 0
                )

                average_candidates = (
                    total_candidates / processed
                )

                average_scored = (
                    total_scored_candidates
                    / processed
                )

                average_matches = (
                    total_matches / processed
                )

                print(
                    f"Processed: "
                    f"{processed:,} | "
                    f"rate={rate:,.1f}/sec | "
                    f"avg candidates="
                    f"{average_candidates:.1f} | "
                    f"avg scored="
                    f"{average_scored:.1f} | "
                    f"avg matches="
                    f"{average_matches:.2f}",
                    flush=True,
                )

    print()

    print(
        "Prediction generation complete."
    )

    print(
        f"Source 1 entities: "
        f"{processed:,}"
    )

    print(
        f"Total blocking candidates: "
        f"{total_candidates:,}"
    )

    print(
        f"Total candidates scored: "
        f"{total_scored_candidates:,}"
    )

    print(
        f"Total predicted matches: "
        f"{total_matches:,}"
    )

    print(
        f"Predicted singletons: "
        f"{singleton_predictions:,}"
    )

    if processed:

        print(
            f"Average candidates per S1: "
            f"{total_candidates / processed:.2f}"
        )

        print(
            f"Average scored candidates per S1: "
            f"{total_scored_candidates / processed:.2f}"
        )

        print(
            f"Average matches per S1: "
            f"{total_matches / processed:.4f}"
        )

        elapsed = time.time() - start

        print(
            f"Total inference time: "
            f"{elapsed:.1f}s"
        )

        print(
            f"Overall rate: "
            f"{processed / elapsed:.2f} Source 1/sec"
        )

    print()

    print(
        f"Matching output: "
        f"{MATCHING_OUTPUT}"
    )

    print(
        f"Candidate output: "
        f"{CANDIDATE_OUTPUT}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "      MEMORY-SAFE ENTITY RESOLUTION INFERENCE"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Load model.
    # --------------------------------------------------------

    print()

    print(
        "Loading trained model..."
    )

    bundle = joblib.load(
        MODEL_PATH
    )

    model = bundle["model"]

    feature_columns = bundle[
        "feature_columns"
    ]

    print(
        "Model loaded."
    )

    print(
        f"Features expected: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Create SQLite database.
    # --------------------------------------------------------

    conn = create_database()

    try:

        # ----------------------------------------------------
        # Source 2.
        # ----------------------------------------------------

        index_source(
            conn,
            TEST_SOURCE2,
            "Source 2",
        )

        # ----------------------------------------------------
        # Source 3.
        # ----------------------------------------------------

        index_source(
            conn,
            TEST_SOURCE3,
            "Source 3",
        )

        # ----------------------------------------------------
        # Create lookup index.
        # ----------------------------------------------------

        finalize_database(
            conn
        )

        # ----------------------------------------------------
        # Generate predictions.
        # ----------------------------------------------------

        process_test_source1(
            conn,
            model,
            feature_columns,
        )

    finally:

        conn.close()

    print()

    print("=" * 70)

    print(
        "INFERENCE COMPLETE"
    )

    print("=" * 70)

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Run the submission validator before uploading."
    )


if __name__ == "__main__":

    main()