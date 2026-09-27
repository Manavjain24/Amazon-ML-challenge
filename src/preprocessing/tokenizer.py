import unicodedata


def tokenize_text(text):
    """
    Unicode-aware tokenizer.

    Keeps letters, numbers, and combining marks together.
    Splits only on punctuation, symbols, and whitespace.
    """

    if not isinstance(text, str):
        return []

    tokens = []
    current = []

    for char in text:
        category = unicodedata.category(char)

        # Letter
        if category.startswith("L"):
            current.append(char)

        # Number
        elif category.startswith("N"):
            current.append(char)

        # Combining mark
        elif category.startswith("M"):
            current.append(char)

        else:
            # Whitespace / punctuation / symbols
            if current:
                tokens.append("".join(current))
                current = []

    if current:
        tokens.append("".join(current))

    return tokens


def tokenize_name(name):
    return tokenize_text(name)


def tokenize_address(address):
    return tokenize_text(address)