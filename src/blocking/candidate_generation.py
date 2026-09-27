```python
from collections import defaultdict

from src.preprocessing.normalization import normalize_name
from src.preprocessing.tokenizer import tokenize_name, tokenize_address


# ============================================================
# CONFIGURATION
# ============================================================

# Maximum number of useful name tokens used for blocking.
MAX_NAME_TOKENS = 5

# Maximum number of useful address tokens used for blocking.
MAX_ADDRESS_TOKENS = 5

# Prefix length used for token-prefix blocking.
TOKEN_PREFIX_LENGTH = 3


# ============================================================
# COMMON / LOW-INFORMATION TOKENS
# ============================================================

# These tokens are common across huge numbers of businesses.
# Using them as blocking keys can create enormous candidate
# blocks and destroy the benefit of blocking.
#
# IMPORTANT:
# These are only ignored for token-based blocking.
# They are NOT removed from the underlying normalized data.

COMMON_NAME_TOKENS = {
    "private",
    "limited",
    "pvt",
    "ltd",
    "llp",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "company",
    "co",
    "enterprises",
    "enterprise",
    "industries",
    "industry",
    "group",
    "holdings",
    "holding",
    "solutions",
    "services",
    "service",
    "international",
    "global",
    "india",
}

COMMON_ADDRESS_TOKENS = {
    "road",
    "rd",
    "street",
    "st",
    "lane",
    "ln",
    "avenue",
    "ave",
    "highway",
    "hwy",
    "building",
    "bldg",
    "floor",
    "fl",
    "shop",
    "office",
    "sector",
    "block",
    "district",
    "city",
    "town",
    "village",
    "state",
    "near",
    "opposite",
    "opp",
    "behind",
    "plot",
    "house",
    "flat",
    "apartment",
    "apt",
    "india",
}


# ============================================================
# TOKEN FILTERING
# ============================================================

def _is_useful_name_token(token):
    """
    Decide whether a name token is useful for blocking.

    The token is kept in the normalized/tokenized data, but
    common generic business/legal words are not used as
    blocking keys.
    """

    if not token:
        return False

    token_lower = token.lower().strip()

    if not token_lower:
        return False

    if token_lower in COMMON_NAME_TOKENS:
        return False

    return True


def _is_useful_address_token(token):
    """
    Decide whether an address token is useful for blocking.

    Generic address words are ignored because they can create
    very large blocks.

    Pure numeric tokens are retained separately because things
    like postal codes, building numbers, plot numbers, etc.
    can be highly discriminative.
    """

    if not token:
        return False

    token_lower = token.lower().strip()

    if not token_lower:
        return False

    if token_lower in COMMON_ADDRESS_TOKENS:
        return False

    return True


def _select_name_tokens(tokens):
    """
    Select a bounded number of informative name tokens.

    We preserve token order but skip generic business/legal
    terms.
    """

    selected = []

    for token in tokens:

        if not _is_useful_name_token(token):
            continue

        if token not in selected:
            selected.append(token)

        if len(selected) >= MAX_NAME_TOKENS:
            break

    return selected


def _select_address_tokens(tokens):
    """
    Select a bounded number of informative address tokens.

    We retain both:
        - informative textual tokens
        - useful numeric tokens

    This is intentionally bounded to prevent candidate explosion.
    """

    selected = []

    for token in tokens:

        if not token:
            continue

        token_lower = token.lower().strip()

        # Keep numeric tokens if they contain useful information.
        if token_lower.isdigit():

            # Ignore extremely short generic numeric values.
            # Two or more digits are generally more useful.
            if len(token_lower) < 2:
                continue

            if token not in selected:
                selected.append(token)

        else:

            if not _is_useful_address_token(token):
                continue

            if token not in selected:
                selected.append(token)

        if len(selected) >= MAX_ADDRESS_TOKENS:
            break

    return selected


# ============================================================
# BLOCKING KEY GENERATION
# ============================================================

def create_block_keys(country, business_name, business_address):
    """
    Generate multiple blocking keys for a record.

    Blocking is intentionally multi-strategy.

    Existing strategies are preserved for compatibility:

        NAME_PREFIX
        NAME_TOKEN
        NAME_TOKEN_PREFIX
        ADDRESS_TOKEN
        ADDRESS_TOKEN_PREFIX

    Improvements:

        1. Name blocking is no longer restricted to the first
           name token.

        2. Address blocking is no longer restricted to the first
           non-numeric address token.

        3. Multiple informative tokens are used.

        4. Generic business/legal/address tokens are excluded
           from token-based blocking.

        5. Useful numeric address tokens are retained.

    Returns:
        set[str]
    """

    keys = set()

    country = str(country).strip()

    normalized_name = normalize_name(
        str(business_name)
    )

    normalized_address = str(
        business_address
    ).strip()

    # ========================================================
    # 1. NAME PREFIX
    # ========================================================

    if normalized_name:

        prefix = normalized_name[:2]

        if prefix:

            keys.add(
                f"NAME_PREFIX|{country}|{prefix}"
            )

    # ========================================================
    # 2. NAME TOKEN BLOCKING
    # ========================================================

    name_tokens = tokenize_name(
        normalized_name
    )

    useful_name_tokens = _select_name_tokens(
        name_tokens
    )

    for token in useful_name_tokens:

        # Full token key.
        keys.add(
            f"NAME_TOKEN|{country}|{token}"
        )

        # Short token-prefix key.
        if len(token) >= TOKEN_PREFIX_LENGTH:

            keys.add(
                f"NAME_TOKEN_PREFIX|{country}|"
                f"{token[:TOKEN_PREFIX_LENGTH]}"
            )

    # ========================================================
    # 3. ADDRESS TOKEN BLOCKING
    # ========================================================

    address_tokens = tokenize_address(
        normalized_address
    )

    useful_address_tokens = _select_address_tokens(
        address_tokens
    )

    for token in useful_address_tokens:

        # Full address token.
        keys.add(
            f"ADDRESS_TOKEN|{country}|{token}"
        )

        # Prefix is useful for spelling variations such as:
        #
        # Wayne
        # Wanye
        #
        # Both can potentially produce:
        #
        # Way
        #
        if len(token) >= TOKEN_PREFIX_LENGTH:

            keys.add(
                f"ADDRESS_TOKEN_PREFIX|{country}|"
                f"{token[:TOKEN_PREFIX_LENGTH]}"
            )

    return keys


# ============================================================
# BUILD BLOCK INDEX
# ============================================================

def build_block_index(source_df):
    """
    Build a multi-key blocking index.

    Example:

        {
            "NAME_PREFIX|US|wi": [10, 25, 100],
            "NAME_TOKEN|US|wilson": [10, 25],
            "NAME_TOKEN_PREFIX|US|wil": [10, 25],
            "ADDRESS_TOKEN|US|texas": [10, 90]
        }

    Each record can belong to multiple blocks.

    The DataFrame index is preserved as the candidate ID so
    existing downstream code remains compatible.
    """

    block_index = defaultdict(list)

    for idx, row in source_df.iterrows():

        keys = create_block_keys(
            row["country"],
            row["business_name"],
            row["business_address"]
        )

        for key in keys:

            block_index[key].append(idx)

    return dict(block_index)


# ============================================================
# CANDIDATE RETRIEVAL
# ============================================================

def get_candidates(source_record, block_index):
    """
    Retrieve the union of candidates from all applicable
    blocking strategies.

    Duplicate candidates are removed.

    The return type remains a list so existing downstream
    pipeline code does not need to change.
    """

    keys = create_block_keys(
        source_record["country"],
        source_record["business_name"],
        source_record["business_address"]
    )

    candidates = set()

    for key in keys:

        for idx in block_index.get(key, []):

            candidates.add(idx)

    return list(candidates)
```
