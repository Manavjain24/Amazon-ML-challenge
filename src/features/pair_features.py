    
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

from src.blocking.candidate_generation import (
    _romanize,
    _consonant_skeleton
)


def _token_overlap_count(tokens_a, tokens_b):
    """
    Count the number of shared unique tokens.
    """
    set_a = set(tokens_a)
    set_b = set(tokens_b)

    return len(set_a.intersection(set_b))


def _length_ratio(text_a, text_b):
    """
    Return the ratio of the shorter non-empty string length
    to the longer string length.

    Returns:
        float between 0.0 and 1.0
    """

    len_a = len(text_a)
    len_b = len(text_b)

    if len_a == 0 or len_b == 0:
        return 0.0

    return min(len_a, len_b) / max(len_a, len_b)


def _transliterated_tokens(text):
    """
    Convert normalized text into romanized tokens.

    The same _romanize() implementation used by blocking
    is deliberately reused here.
    """

    romanized = _romanize(text)

    if not romanized:
        return []

    return romanized.split()


def _skeleton_tokens(text):
    """
    Convert normalized text into consonant-skeleton tokens.

    The same _consonant_skeleton() implementation used by
    blocking is deliberately reused here.
    """

    skeleton = _consonant_skeleton(text)

    if not skeleton:
        return []

    return skeleton.split()


def extract_pair_features(source1_record, source2_record):
    """
    Extract similarity features from a Source 1 / Source 2
    candidate pair.

    The function is source-agnostic despite the historical
    function name: source2_record can represent either a
    Source 2 or Source 3 record.

    Returns:
        dict[str, float]
    """

    # ========================================================
    # NORMALIZE
    # ========================================================

    name1 = normalize_name(
        str(source1_record.get("business_name", ""))
    )

    name2 = normalize_name(
        str(source2_record.get("business_name", ""))
    )

    address1 = normalize_address(
        str(source1_record.get("business_address", ""))
    )

    address2 = normalize_address(
        str(source2_record.get("business_address", ""))
    )

    # ========================================================
    # TOKENIZE
    # ========================================================

    name_tokens1 = tokenize_name(name1)
    name_tokens2 = tokenize_name(name2)

    address_tokens1 = tokenize_address(address1)
    address_tokens2 = tokenize_address(address2)

    # ========================================================
    # ORIGINAL NAME FEATURES
    # ========================================================

    name_jaccard = jaccard_similarity(
        name_tokens1,
        name_tokens2
    )

    name_levenshtein = levenshtein_similarity(
        name1,
        name2
    )

    # ========================================================
    # ORIGINAL ADDRESS FEATURES
    # ========================================================

    address_jaccard = jaccard_similarity(
        address_tokens1,
        address_tokens2
    )

    address_levenshtein = levenshtein_similarity(
        address1,
        address2
    )

    # ========================================================
    # TOKEN OVERLAP
    # ========================================================

    name_token_overlap = _token_overlap_count(
        name_tokens1,
        name_tokens2
    )

    address_token_overlap = _token_overlap_count(
        address_tokens1,
        address_tokens2
    )

    # ========================================================
    # TRANSLITERATION FEATURES
    # ========================================================

    translit_tokens1 = _transliterated_tokens(name1)
    translit_tokens2 = _transliterated_tokens(name2)

    name_translit_jaccard = jaccard_similarity(
        translit_tokens1,
        translit_tokens2
    )

    translit_name1 = " ".join(translit_tokens1)
    translit_name2 = " ".join(translit_tokens2)

    name_translit_levenshtein = levenshtein_similarity(
        translit_name1,
        translit_name2
    )

    # ========================================================
    # CONSONANT SKELETON FEATURES
    # ========================================================

    skeleton_tokens1 = _skeleton_tokens(name1)
    skeleton_tokens2 = _skeleton_tokens(name2)

    name_skeleton_jaccard = jaccard_similarity(
        skeleton_tokens1,
        skeleton_tokens2
    )

    skeleton_name1 = " ".join(skeleton_tokens1)
    skeleton_name2 = " ".join(skeleton_tokens2)

    name_skeleton_levenshtein = levenshtein_similarity(
        skeleton_name1,
        skeleton_name2
    )

    # ========================================================
    # EXACT MATCH FEATURES
    # ========================================================

    name_exact = int(
        bool(name1) and
        bool(name2) and
        name1 == name2
    )

    address_exact = int(
        bool(address1) and
        bool(address2) and
        address1 == address2
    )

    # ========================================================
    # LENGTH FEATURES
    # ========================================================

    name_length_ratio = _length_ratio(
        name1,
        name2
    )

    address_length_ratio = _length_ratio(
        address1,
        address2
    )

    # ========================================================
    # COUNTRY
    # ========================================================

    country_match = int(
        str(source1_record.get("country", "")).strip()
        ==
        str(source2_record.get("country", "")).strip()
    )

    # ========================================================
    # RETURN FEATURES
    # ========================================================

    return {
        # Original features
        "name_jaccard": name_jaccard,
        "name_levenshtein": name_levenshtein,
        "address_jaccard": address_jaccard,
        "address_levenshtein": address_levenshtein,
        "country_match": country_match,

        # Token overlap
        "name_token_overlap": name_token_overlap,
        "address_token_overlap": address_token_overlap,

        # Cross-script name features
        "name_translit_jaccard": name_translit_jaccard,
        "name_translit_levenshtein": name_translit_levenshtein,
        "name_skeleton_jaccard": name_skeleton_jaccard,
        "name_skeleton_levenshtein": name_skeleton_levenshtein,

        # Exact matches
        "name_exact": name_exact,
        "address_exact": address_exact,

        # Length features
        "name_length_ratio": name_length_ratio,
        "address_length_ratio": address_length_ratio,
    }

