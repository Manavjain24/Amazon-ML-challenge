
from rapidfuzz.distance import Levenshtein


def jaccard_similarity(tokens_a, tokens_b):
    """
    Calculate Jaccard similarity between two token collections.

    Returns:
        float: value between 0.0 and 1.0
    """

    set_a = set(tokens_a)
    set_b = set(tokens_b)

    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)

    if not union:
        return 0.0

    return len(intersection) / len(union)


def levenshtein_similarity(text_a, text_b):
    """
    Calculate normalized Levenshtein similarity using RapidFuzz.

    Returns:
        float: value between 0.0 and 1.0

    Behavior is kept compatible with the previous implementation:
        - identical strings -> 1.0
        - either empty string -> 0.0
        - otherwise -> normalized similarity
    """

    if text_a == text_b:
        return 1.0

    if not text_a or not text_b:
        return 0.0

    return Levenshtein.normalized_similarity(
        str(text_a),
        str(text_b)
    )

