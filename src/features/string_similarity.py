def jaccard_similarity(tokens_a, tokens_b):
    set_a = set(tokens_a)
    set_b = set(tokens_b)

    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)

    if not union:
        return 0.0

    return len(intersection) / len(union)

def levenshtein_similarity(text_a, text_b):
    if text_a == text_b:
        return 1.0

    if not text_a or not text_b:
        return 0.0

    previous_row = list(range(len(text_b) + 1))

    for i, char_a in enumerate(text_a, start=1):
        current_row = [i]

        for j, char_b in enumerate(text_b, start=1):
            insertion = current_row[j - 1] + 1
            deletion = previous_row[j] + 1
            substitution = previous_row[j - 1] + (char_a != char_b)

            current_row.append(
                min(insertion, deletion, substitution)
            )

        previous_row = current_row

    distance = previous_row[-1]
    max_length = max(len(text_a), len(text_b))

    return 1 - (distance / max_length)