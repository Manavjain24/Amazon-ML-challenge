
import re
import unicodedata
from difflib import SequenceMatcher

import pandas as pd

from src.blocking.candidate_generation import create_block_keys


# ============================================================
# CONFIGURATION
# ============================================================

GROUND_TRUTH_FILE = (
    "data/dataset/train/train_ground_truth.tsv"
)

SOURCE1_FILE = (
    "data/dataset/train/train_source1.tsv"
)

SOURCE2_FILE = (
    "data/dataset/train/train_source2.tsv"
)

SOURCE3_FILE = (
    "data/dataset/train/train_source3.tsv"
)

# Keep this limited while experimenting.
GT_ROWS_TO_TEST = 10000

# Number of detailed missed examples to print.
MISSED_EXAMPLES_TO_PRINT = 10


# ============================================================
# TEXT HELPERS
# ============================================================

def normalize_for_analysis(value):
    """
    Lightweight normalization used ONLY for miss analysis.

    This does NOT change candidate-generation logic.
    """

    if pd.isna(value):
        return ""

    value = str(value).strip().lower()

    # Normalize Unicode representation.
    value = unicodedata.normalize("NFKC", value)

    # Replace punctuation with spaces.
    value = re.sub(
        r"[^\w\s]",
        " ",
        value,
        flags=re.UNICODE
    )

    # Collapse whitespace.
    value = re.sub(
        r"\s+",
        " ",
        value
    ).strip()

    return value


def get_tokens(value):
    """
    Return whitespace-separated normalized tokens.
    """

    normalized = normalize_for_analysis(value)

    if not normalized:
        return set()

    return set(normalized.split())


def shared_tokens(value1, value2):
    """
    Return tokens appearing in both values.
    """

    tokens1 = get_tokens(value1)
    tokens2 = get_tokens(value2)

    return tokens1 & tokens2


def character_similarity(value1, value2):
    """
    SequenceMatcher similarity.

    Used ONLY for diagnostics.
    It does NOT participate in blocking.
    """

    value1 = normalize_for_analysis(value1)
    value2 = normalize_for_analysis(value2)

    if not value1 or not value2:
        return 0.0

    return SequenceMatcher(
        None,
        value1,
        value2
    ).ratio()


def get_scripts(value):
    """
    Rough Unicode script detection used only for diagnostics.
    """

    scripts = set()

    for char in str(value):

        if char.isspace():
            continue

        name = unicodedata.name(char, "")

        if "LATIN" in name:
            scripts.add("LATIN")

        elif "DEVANAGARI" in name:
            scripts.add("DEVANAGARI")

        elif "TAMIL" in name:
            scripts.add("TAMIL")

        elif "GUJARATI" in name:
            scripts.add("GUJARATI")

        elif "KANNADA" in name:
            scripts.add("KANNADA")

        elif "TELUGU" in name:
            scripts.add("TELUGU")

        elif "BENGALI" in name:
            scripts.add("BENGALI")

        elif "MALAYALAM" in name:
            scripts.add("MALAYALAM")

        elif "GURMUKHI" in name:
            scripts.add("GURMUKHI")

        elif "ORIYA" in name:
            scripts.add("ORIYA")

        elif "ARABIC" in name:
            scripts.add("ARABIC")

        elif "CYRILLIC" in name:
            scripts.add("CYRILLIC")

        elif "GREEK" in name:
            scripts.add("GREEK")

        elif char.isdigit():
            scripts.add("DIGIT")

    return scripts


def is_different_script(value1, value2):
    """
    True when both values contain useful scripts and
    their script sets have no overlap.
    """

    scripts1 = get_scripts(value1)
    scripts2 = get_scripts(value2)

    if not scripts1 or not scripts2:
        return False

    return scripts1.isdisjoint(scripts2)


# ============================================================
# GROUND TRUTH HELPERS
# ============================================================

def get_required_entity_ids(ground_truth_df):
    """
    Extract all Source 1, Source 2 and Source 3 IDs
    required by the selected ground-truth sample.
    """

    source1_ids = set()
    source2_ids = set()
    source3_ids = set()

    for _, row in ground_truth_df.iterrows():

        source1_id = str(
            row["source1_entity_id"]
        ).strip()

        if source1_id:
            source1_ids.add(source1_id)

        matched_ids = row["matched_entity_ids"]

        if pd.isna(matched_ids):
            continue

        for entity_id in str(
            matched_ids
        ).split(","):

            entity_id = entity_id.strip()

            if not entity_id:
                continue

            if entity_id.startswith("S2-"):
                source2_ids.add(entity_id)

            elif entity_id.startswith("S3-"):
                source3_ids.add(entity_id)

    return (
        source1_ids,
        source2_ids,
        source3_ids
    )


def count_ground_truth_relationships(ground_truth_df):
    """
    Count every individual GT relationship.

    Example:

        S1-A -> S2-X,S3-Y,S3-Z

    counts as 3 relationships.
    """

    total = 0

    for matched_ids in ground_truth_df[
        "matched_entity_ids"
    ]:

        if pd.isna(matched_ids):
            continue

        for entity_id in str(
            matched_ids
        ).split(","):

            if entity_id.strip():
                total += 1

    return total


# ============================================================
# LOAD REQUIRED SOURCE 2 / SOURCE 3 RECORDS
# ============================================================

def load_required_records(
    filepath,
    required_ids,
    chunksize=100000
):
    """
    Load only records whose entity_id is required
    by the current GT sample.
    """

    records = {}

    if not required_ids:
        return records

    for chunk in pd.read_csv(
        filepath,
        sep="\t",
        chunksize=chunksize
    ):

        matched = chunk[
            chunk["entity_id"].isin(required_ids)
        ]

        for row in matched.itertuples(
            index=False
        ):

            records[row.entity_id] = row

        if len(records) == len(required_ids):
            break

    return records


# ============================================================
# BUILD SOURCE 1 BLOCK LOOKUP
# ============================================================

def build_source1_block_lookup(source1_df):
    """
    Build:

        entity_id -> set(block_keys)
    """

    block_lookup = {}

    for row in source1_df.itertuples(
        index=False
    ):

        keys = create_block_keys(
            row.country,
            row.business_name,
            row.business_address
        )

        block_lookup[
            row.entity_id
        ] = keys

    return block_lookup


# ============================================================
# BUILD TARGET BLOCK LOOKUP
# ============================================================

def build_target_block_lookup(records):
    """
    Build:

        entity_id -> set(block_keys)
    """

    block_lookup = {}

    for entity_id, row in records.items():

        keys = create_block_keys(
            row.country,
            row.business_name,
            row.business_address
        )

        block_lookup[
            entity_id
        ] = keys

    return block_lookup


# ============================================================
# MISS ANALYSIS
# ============================================================

def analyze_missed_match(
    source1_row,
    target_row,
    source1_keys,
    target_keys
):
    """
    Analyze one true relationship that was missed.

    This function does NOT modify candidate generation.
    """

    source1_name = source1_row["business_name"]
    target_name = target_row.business_name

    source1_address = source1_row["business_address"]
    target_address = target_row.business_address

    name_shared = shared_tokens(
        source1_name,
        target_name
    )

    address_shared = shared_tokens(
        source1_address,
        target_address
    )

    name_similarity = character_similarity(
        source1_name,
        target_name
    )

    address_similarity = character_similarity(
        source1_address,
        target_address
    )

    different_name_script = is_different_script(
        source1_name,
        target_name
    )

    different_address_script = is_different_script(
        source1_address,
        target_address
    )

    return {
        "name_shared_tokens": name_shared,
        "address_shared_tokens": address_shared,
        "name_similarity": name_similarity,
        "address_similarity": address_similarity,
        "different_name_script": different_name_script,
        "different_address_script": different_address_script,
        "source1_name_scripts": get_scripts(
            source1_name
        ),
        "target_name_scripts": get_scripts(
            target_name
        ),
        "source1_address_scripts": get_scripts(
            source1_address
        ),
        "target_address_scripts": get_scripts(
            target_address
        ),
        "source1_keys": source1_keys,
        "target_keys": target_keys,
    }


# ============================================================
# MAIN
# ============================================================

print("Loading ground truth...")

ground_truth_df = pd.read_csv(
    GROUND_TRUTH_FILE,
    sep="\t",
    nrows=GT_ROWS_TO_TEST
)

print(
    f"Ground truth rows: "
    f"{len(ground_truth_df):,}"
)


# ============================================================
# GROUND TRUTH COUNTS
# ============================================================

(
    required_source1_ids,
    required_source2_ids,
    required_source3_ids
) = get_required_entity_ids(
    ground_truth_df
)

total_ground_truth_matches = (
    count_ground_truth_relationships(
        ground_truth_df
    )
)

print(
    f"Required Source 1 records: "
    f"{len(required_source1_ids):,}"
)

print(
    f"Required Source 2 records: "
    f"{len(required_source2_ids):,}"
)

print(
    f"Required Source 3 records: "
    f"{len(required_source3_ids):,}"
)

print(
    f"Total GT relationships: "
    f"{total_ground_truth_matches:,}"
)


# ============================================================
# LOAD SOURCE 1
# ============================================================

print()
print("Loading Source 1...")

source1_df = pd.read_csv(
    SOURCE1_FILE,
    sep="\t"
)

print(
    f"Source 1 records: "
    f"{len(source1_df):,}"
)


# ============================================================
# BUILD SOURCE 1 BLOCK LOOKUP
# ============================================================

print()
print("Building Source 1 block index...")

source1_block_lookup = (
    build_source1_block_lookup(
        source1_df
    )
)

print(
    f"Number of Source 1 records indexed: "
    f"{len(source1_block_lookup):,}"
)


# ============================================================
# IMPORTANT PERFORMANCE FIX
# ============================================================
#
# Previously, inside the GT loop we did:
#
#     source1_df[
#         source1_df["entity_id"] == source1_id
#     ]
#
# That scans 2.2M Source 1 rows for EVERY GT relationship.
#
# Instead, build a lookup ONLY for the Source 1 records
# actually required by this GT sample.
#
# This turns the repeated O(2.2M) lookup into O(1).
# ============================================================

print()
print("Building Source 1 row lookup for GT sample...")

required_source1_df = source1_df[
    source1_df["entity_id"].isin(
        required_source1_ids
    )
]

source1_row_lookup = (
    required_source1_df
    .set_index("entity_id")
    .to_dict("index")
)

print(
    f"Source 1 GT rows available: "
    f"{len(source1_row_lookup):,}"
)


missing_source1_records = (
    required_source1_ids
    - set(source1_row_lookup.keys())
)

if missing_source1_records:

    print(
        f"WARNING: "
        f"{len(missing_source1_records):,} "
        f"required Source 1 records were not found."
    )


# ============================================================
# LOAD REQUIRED SOURCE 2
# ============================================================

print()
print("Loading required Source 2 records...")

source2_records = load_required_records(
    SOURCE2_FILE,
    required_source2_ids
)

print(
    f"Required Source 2 records loaded: "
    f"{len(source2_records):,}"
)

missing_source2_records = (
    required_source2_ids
    - set(source2_records.keys())
)

if missing_source2_records:

    print(
        f"WARNING: "
        f"{len(missing_source2_records):,} "
        f"required Source 2 records were not found."
    )


# ============================================================
# BUILD SOURCE 2 BLOCK LOOKUP
# ============================================================

print()
print("Building Source 2 blocking keys...")

source2_block_lookup = (
    build_target_block_lookup(
        source2_records
    )
)


# ============================================================
# LOAD REQUIRED SOURCE 3
# ============================================================

print()
print("Loading required Source 3 records...")

source3_records = load_required_records(
    SOURCE3_FILE,
    required_source3_ids
)

print(
    f"Required Source 3 records loaded: "
    f"{len(source3_records):,}"
)

missing_source3_records = (
    required_source3_ids
    - set(source3_records.keys())
)

if missing_source3_records:

    print(
        f"WARNING: "
        f"{len(missing_source3_records):,} "
        f"required Source 3 records were not found."
    )


# ============================================================
# BUILD SOURCE 3 BLOCK LOOKUP
# ============================================================

print()
print("Building Source 3 blocking keys...")

source3_block_lookup = (
    build_target_block_lookup(
        source3_records
    )
)


# ============================================================
# TEST BLOCKING RECALL
# ============================================================

print()
print("Testing blocking recall...")

found_matches = 0
missed_matches = 0
unavailable_matches = 0

missed_examples = []


# ============================================================
# STRATEGY HIT COUNTS
# ============================================================

strategy_hits = {
    "NAME_PREFIX": 0,
    "NAME_TOKEN_PREFIX": 0,
    "NAME_TOKEN": 0,
    "ADDRESS_TOKEN": 0,
    "ADDRESS_TOKEN_PREFIX": 0,
}


# ============================================================
# MISS ANALYSIS COUNTERS
# ============================================================

miss_analysis = {
    "name_shared_tokens": 0,
    "address_shared_tokens": 0,
    "name_or_address_overlap": 0,
    "no_token_overlap": 0,

    "high_name_similarity": 0,
    "high_address_similarity": 0,

    "different_name_script": 0,
    "different_address_script": 0,
}

name_similarity_values = []
address_similarity_values = []


# ============================================================
# EVALUATION
# ============================================================

for _, gt_row in ground_truth_df.iterrows():

    source1_id = str(
        gt_row["source1_entity_id"]
    ).strip()

    # --------------------------------------------------------
    # SOURCE 1 BLOCK KEYS
    # --------------------------------------------------------

    source1_keys = source1_block_lookup.get(
        source1_id
    )

    if source1_keys is None:

        print(
            f"WARNING: Source 1 record not found: "
            f"{source1_id}"
        )

        # Every GT relationship attached to this Source 1
        # record is unavailable for evaluation.
        matched_ids = gt_row[
            "matched_entity_ids"
        ]

        if not pd.isna(matched_ids):

            for matched_id in str(
                matched_ids
            ).split(","):

                if matched_id.strip():
                    unavailable_matches += 1

        continue


    # --------------------------------------------------------
    # O(1) SOURCE 1 ROW LOOKUP
    # --------------------------------------------------------
    #
    # This replaces the old expensive:
    #
    # source1_df[
    #     source1_df["entity_id"] == source1_id
    # ]
    #
    # --------------------------------------------------------

    source1_row = source1_row_lookup.get(
        source1_id
    )

    if source1_row is None:

        matched_ids = gt_row[
            "matched_entity_ids"
        ]

        if not pd.isna(matched_ids):

            for matched_id in str(
                matched_ids
            ).split(","):

                if matched_id.strip():
                    unavailable_matches += 1

        continue


    # --------------------------------------------------------
    # GET GROUND TRUTH MATCHES
    # --------------------------------------------------------

    matched_ids = gt_row[
        "matched_entity_ids"
    ]

    if pd.isna(matched_ids):
        continue

    matched_ids = str(
        matched_ids
    ).split(",")


    # --------------------------------------------------------
    # EVALUATE EVERY TRUE RELATIONSHIP
    # --------------------------------------------------------

    for matched_id in matched_ids:

        matched_id = matched_id.strip()

        if not matched_id:
            continue


        # ====================================================
        # SELECT TARGET LOOKUP
        # ====================================================

        if matched_id.startswith("S2-"):

            target_lookup = (
                source2_block_lookup
            )

            target_records = (
                source2_records
            )

        elif matched_id.startswith("S3-"):

            target_lookup = (
                source3_block_lookup
            )

            target_records = (
                source3_records
            )

        else:

            print(
                f"WARNING: Unknown entity ID format: "
                f"{matched_id}"
            )

            unavailable_matches += 1

            continue


        # ====================================================
        # CHECK WHETHER TARGET RECORD WAS LOADED
        # ====================================================

        if matched_id not in target_lookup:

            unavailable_matches += 1

            continue


        # ====================================================
        # GET TARGET BLOCK KEYS
        # ====================================================

        target_keys = target_lookup[
            matched_id
        ]


        # ====================================================
        # CHECK BLOCK OVERLAP
        # ====================================================

        matched = False
        matched_strategy = None

        for key in source1_keys:

            if key in target_keys:

                matched = True

                matched_strategy = (
                    key.split("|")[0]
                )

                break


        # ====================================================
        # CLASSIFY RESULT
        # ====================================================

        if matched:

            found_matches += 1

            if matched_strategy in strategy_hits:

                strategy_hits[
                    matched_strategy
                ] += 1

            continue


        # ====================================================
        # MISSED MATCH
        # ====================================================

        missed_matches += 1

        target_row = target_records[
            matched_id
        ]


        # ====================================================
        # MISS ANALYSIS
        # ====================================================

        analysis = analyze_missed_match(
            source1_row,
            target_row,
            source1_keys,
            target_keys
        )


        # ----------------------------------------------------
        # TOKEN OVERLAP
        # ----------------------------------------------------

        has_name_overlap = bool(
            analysis[
                "name_shared_tokens"
            ]
        )

        has_address_overlap = bool(
            analysis[
                "address_shared_tokens"
            ]
        )

        if has_name_overlap:

            miss_analysis[
                "name_shared_tokens"
            ] += 1

        if has_address_overlap:

            miss_analysis[
                "address_shared_tokens"
            ] += 1

        if (
            has_name_overlap
            or has_address_overlap
        ):

            miss_analysis[
                "name_or_address_overlap"
            ] += 1

        if (
            not has_name_overlap
            and not has_address_overlap
        ):

            miss_analysis[
                "no_token_overlap"
            ] += 1


        # ----------------------------------------------------
        # SIMILARITY
        # ----------------------------------------------------

        name_similarity = analysis[
            "name_similarity"
        ]

        address_similarity = analysis[
            "address_similarity"
        ]

        name_similarity_values.append(
            name_similarity
        )

        address_similarity_values.append(
            address_similarity
        )

        if name_similarity >= 0.80:

            miss_analysis[
                "high_name_similarity"
            ] += 1

        if address_similarity >= 0.80:

            miss_analysis[
                "high_address_similarity"
            ] += 1


        # ----------------------------------------------------
        # SCRIPT DIFFERENCE
        # ----------------------------------------------------

        if analysis[
            "different_name_script"
        ]:

            miss_analysis[
                "different_name_script"
            ] += 1

        if analysis[
            "different_address_script"
        ]:

            miss_analysis[
                "different_address_script"
            ] += 1


        # ----------------------------------------------------
        # STORE EXAMPLES
        # ----------------------------------------------------

        if len(missed_examples) < (
            MISSED_EXAMPLES_TO_PRINT
        ):

            missed_examples.append(
                {
                    "source1_id":
                        source1_id,

                    "matched_id":
                        matched_id,

                    "source1_name":
                        source1_row[
                            "business_name"
                        ],

                    "target_name":
                        target_row.business_name,

                    "source1_address":
                        source1_row[
                            "business_address"
                        ],

                    "target_address":
                        target_row.business_address,

                    "source1_keys":
                        source1_keys,

                    "target_keys":
                        target_keys,

                    "name_shared_tokens":
                        analysis[
                            "name_shared_tokens"
                        ],

                    "address_shared_tokens":
                        analysis[
                            "address_shared_tokens"
                        ],

                    "name_similarity":
                        analysis[
                            "name_similarity"
                        ],

                    "address_similarity":
                        analysis[
                            "address_similarity"
                        ],

                    "different_name_script":
                        analysis[
                            "different_name_script"
                        ],

                    "different_address_script":
                        analysis[
                            "different_address_script"
                        ],
                }
            )


# ============================================================
# FINAL ACCOUNTING
# ============================================================

evaluated_matches = (
    found_matches
    + missed_matches
)

# The true GT total is calculated independently above.
#
# This is important:
#
#     total GT = evaluated + unavailable
#
# We do NOT derive total GT from evaluated results.
# ============================================================

accounted_ground_truth = (
    evaluated_matches
    + unavailable_matches
)


# ============================================================
# SANITY CHECK
# ============================================================

assert (
    found_matches
    + missed_matches
    == evaluated_matches
), (
    "Evaluation accounting error: "
    f"found={found_matches}, "
    f"missed={missed_matches}, "
    f"evaluated={evaluated_matches}"
)

assert (
    accounted_ground_truth
    == total_ground_truth_matches
), (
    "Total ground-truth accounting error: "
    f"evaluated={evaluated_matches}, "
    f"unavailable={unavailable_matches}, "
    f"accounted={accounted_ground_truth}, "
    f"actual_gt={total_ground_truth_matches}"
)


# ============================================================
# RECALL
# ============================================================

if evaluated_matches > 0:

    blocking_recall = (
        found_matches
        / evaluated_matches
    ) * 100

else:

    blocking_recall = 0.0


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 55)
print("              BLOCKING RECALL RESULTS")
print("=" * 55)

print(
    f"Ground truth rows tested: "
    f"{len(ground_truth_df):,}"
)

print(
    f"Total GT relationships: "
    f"{total_ground_truth_matches:,}"
)

print(
    f"Evaluated relationships: "
    f"{evaluated_matches:,}"
)

print(
    f"Found: "
    f"{found_matches:,}"
)

print(
    f"Missed: "
    f"{missed_matches:,}"
)

print(
    f"Unavailable: "
    f"{unavailable_matches:,}"
)

print(
    f"Blocking Recall: "
    f"{blocking_recall:.2f}%"
)


# ============================================================
# ACCOUNTING CHECK
# ============================================================

print()
print("=" * 55)
print("                 ACCOUNTING CHECK")
print("=" * 55)

print(
    f"{found_matches:,} found + "
    f"{missed_matches:,} missed = "
    f"{evaluated_matches:,} evaluated"
)

print(
    f"{evaluated_matches:,} evaluated + "
    f"{unavailable_matches:,} unavailable = "
    f"{accounted_ground_truth:,} accounted"
)

print(
    f"Actual GT relationships = "
    f"{total_ground_truth_matches:,}"
)


# ============================================================
# STRATEGY HITS
# ============================================================

print()
print("=" * 55)
print("              BLOCKING STRATEGY HITS")
print("=" * 55)

for strategy, count in strategy_hits.items():

    print(
        f"{strategy}: {count:,}"
    )


# ============================================================
# MISSED MATCH ANALYSIS
# ============================================================

print()
print("=" * 55)
print("              MISSED MATCH ANALYSIS")
print("=" * 55)

print(
    f"Total missed relationships: "
    f"{missed_matches:,}"
)

if missed_matches > 0:

    print()
    print("TOKEN OVERLAP")
    print("-" * 55)

    print(
        f"Name has shared tokens: "
        f"{miss_analysis['name_shared_tokens']:,}"
    )

    print(
        f"Address has shared tokens: "
        f"{miss_analysis['address_shared_tokens']:,}"
    )

    print(
        f"Name OR address has shared tokens: "
        f"{miss_analysis['name_or_address_overlap']:,}"
    )

    print(
        f"No shared name/address tokens: "
        f"{miss_analysis['no_token_overlap']:,}"
    )


    print()
    print("CHARACTER SIMILARITY")
    print("-" * 55)

    print(
        f"Name similarity >= 0.80: "
        f"{miss_analysis['high_name_similarity']:,}"
    )

    print(
        f"Address similarity >= 0.80: "
        f"{miss_analysis['high_address_similarity']:,}"
    )


    print()
    print("SCRIPT DIFFERENCES")
    print("-" * 55)

    print(
        f"Different name scripts: "
        f"{miss_analysis['different_name_script']:,}"
    )

    print(
        f"Different address scripts: "
        f"{miss_analysis['different_address_script']:,}"
    )


    print()
    print("AVERAGE SIMILARITY")
    print("-" * 55)

    if name_similarity_values:

        average_name_similarity = (
            sum(name_similarity_values)
            / len(name_similarity_values)
        )

        print(
            f"Average missed-name similarity: "
            f"{average_name_similarity:.3f}"
        )

    if address_similarity_values:

        average_address_similarity = (
            sum(address_similarity_values)
            / len(address_similarity_values)
        )

        print(
            f"Average missed-address similarity: "
            f"{average_address_similarity:.3f}"
        )


# ============================================================
# MISSED MATCH EXAMPLES
# ============================================================

print()
print("=" * 55)
print("             MISSED MATCH EXAMPLES")
print("=" * 55)

if not missed_examples:

    print("No missed matches.")

else:

    for example in missed_examples:

        print()

        print(
            f"Source 1: "
            f"{example['source1_id']}"
        )

        print(
            f"True match: "
            f"{example['matched_id']}"
        )

        print(
            f"Source 1 name: "
            f"{example['source1_name']}"
        )

        print(
            f"True match name: "
            f"{example['target_name']}"
        )

        print(
            f"Source 1 address: "
            f"{example['source1_address']}"
        )

        print(
            f"True match address: "
            f"{example['target_address']}"
        )

        print(
            f"Shared name tokens: "
            f"{example['name_shared_tokens']}"
        )

        print(
            f"Shared address tokens: "
            f"{example['address_shared_tokens']}"
        )

        print(
            f"Name similarity: "
            f"{example['name_similarity']:.3f}"
        )

        print(
            f"Address similarity: "
            f"{example['address_similarity']:.3f}"
        )

        print(
            f"Different name script: "
            f"{example['different_name_script']}"
        )

        print(
            f"Different address script: "
            f"{example['different_address_script']}"
        )

        print(
            f"Source 1 keys: "
            f"{example['source1_keys']}"
        )

        print(
            f"True match keys: "
            f"{example['target_keys']}"
        )
