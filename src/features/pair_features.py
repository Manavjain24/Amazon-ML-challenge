from src.preprocessing.normalization import (
    normalize_name,
    normalize_address
)

from src.preprocessing.tokenizer import (
    tokenize_name,
    tokenize_address
)

from src.features.string_similarity import (
    jaccard_similarity,
    levenshtein_similarity
)


def extract_pair_features(source1_record, source2_record):
    """
    Extract similarity features from a Source 1 / Source 2 pair.
    """

    # -----------------------------
    # Normalize
    # -----------------------------

    name1 = normalize_name(source1_record["business_name"])
    name2 = normalize_name(source2_record["business_name"])

    address1 = normalize_address(
        source1_record["business_address"]
    )

    address2 = normalize_address(
        source2_record["business_address"]
    )

    # -----------------------------
    # Tokenize
    # -----------------------------

    name_tokens1 = tokenize_name(name1)
    name_tokens2 = tokenize_name(name2)

    address_tokens1 = tokenize_address(address1)
    address_tokens2 = tokenize_address(address2)

    # -----------------------------
    # Similarity features
    # -----------------------------

    name_jaccard = jaccard_similarity(
        name_tokens1,
        name_tokens2
    )

    address_jaccard = jaccard_similarity(
        address_tokens1,
        address_tokens2
    )

    name_levenshtein = levenshtein_similarity(
        name1,
        name2
    )

    address_levenshtein = levenshtein_similarity(
        address1,
        address2
    )

    # -----------------------------
    # Exact country match
    # -----------------------------

    country_match = int(
        source1_record["country"] ==
        source2_record["country"]
    )

    return {
        "name_jaccard": name_jaccard,
        "name_levenshtein": name_levenshtein,
        "address_jaccard": address_jaccard,
        "address_levenshtein": address_levenshtein,
        "country_match": country_match
    }