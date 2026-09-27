
from collections import defaultdict
import re
import unicodedata

try:
    from unidecode import unidecode
except ImportError:
    unidecode = None

from src.preprocessing.normalization import normalize_name
from src.preprocessing.tokenizer import tokenize_name, tokenize_address


# ============================================================
# CONFIGURATION
# ============================================================

MAX_NAME_TOKENS = 5
MAX_ADDRESS_TOKENS = 5

TOKEN_PREFIX_LENGTH = 3

# Transliteration blocking is intentionally limited.
#
# We do NOT replace the original Unicode representation.
# These are additional candidate-generation keys only.
TRANSLIT_TOKEN_PREFIX_LENGTH = 3
SKELETON_PREFIX_LENGTH = 2


# ============================================================
# COMMON / LOW-INFORMATION TOKENS
# ============================================================

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
    Return True when a name token contains useful business
    identity information.

    Generic legal/business words are excluded from blocking
    keys, but remain untouched in the original data.
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
    Return True when an address token is potentially useful
    for blocking.
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
    Select up to MAX_NAME_TOKENS informative name tokens.
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
    Select up to MAX_ADDRESS_TOKENS useful address tokens.

    Numeric tokens are retained because PIN codes, house
    numbers, plot numbers, etc. can be highly discriminative.
    """

    selected = []

    for token in tokens:

        if not token:
            continue

        token_lower = token.lower().strip()

        if token_lower.isdigit():

            # Avoid single-digit noise.
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
# TRANSLITERATION HELPERS
# ============================================================

def _romanize(text):
    """
    Produce a rough Latin/roman representation of Unicode text.

    This is deliberately used ONLY for blocking.

    The original Unicode text remains the canonical value used
    by the rest of the matching pipeline.

    If Unidecode is not installed, return an empty string so
    the existing Unicode blocker continues to work unchanged.
    """

    if not text:
        return ""

    if unidecode is None:
        return ""

    try:
        value = unidecode(text)
    except Exception:
        return ""

    value = value.lower()

    # Keep only ASCII letters and digits.
    value = re.sub(r"[^a-z0-9]+", " ", value)

    # Collapse whitespace.
    value = re.sub(r"\s+", " ", value).strip()

    return value


def _consonant_skeleton(text):
    """
    Create a coarse consonant representation of romanized text.

    Example:

        "sun"  -> "sn"
        "सन"   -> roughly "sn"

    This is intentionally coarse.

    It is NOT used as a similarity score.
    It is only an additional blocking key to bridge
    different writing systems.
    """

    if not text:
        return ""

    romanized = _romanize(text)

    if not romanized:
        return ""

    skeleton_tokens = []

    for token in romanized.split():

        # Keep digits intact.
        if token.isdigit():
            skeleton_tokens.append(token)
            continue

        consonants = "".join(
            char
            for char in token
            if char.isalpha()
            and char not in "aeiou"
        )

        if consonants:
            skeleton_tokens.append(consonants)

    return " ".join(skeleton_tokens)


def _transliteration_variants(token):
    """
    Generate conservative blocking variants for one token.

    Variants:

        1. Full romanized token
        2. Romanized prefix
        3. Consonant skeleton
        4. Consonant skeleton prefix

    These are ADDITIONAL blocking keys.

    The original Unicode token is always handled separately.
    """

    variants = set()

    if not token:
        return variants

    romanized = _romanize(token)

    if romanized:

        # Usually romanization of one token stays one token,
        # but take the first component if punctuation causes
        # multiple pieces.
        roman_tokens = romanized.split()

        for roman_token in roman_tokens:

            if not roman_token:
                continue

            variants.add(
                ("FULL", roman_token)
            )

            if len(roman_token) >= TRANSLIT_TOKEN_PREFIX_LENGTH:

                variants.add(
                    (
                        "PREFIX",
                        roman_token[:TRANSLIT_TOKEN_PREFIX_LENGTH]
                    )
                )

    skeleton = _consonant_skeleton(token)

    if skeleton:

        skeleton_tokens = skeleton.split()

        for skeleton_token in skeleton_tokens:

            if not skeleton_token:
                continue

            variants.add(
                ("SKELETON", skeleton_token)
            )

            if len(skeleton_token) >= SKELETON_PREFIX_LENGTH:

                variants.add(
                    (
                        "SKELETON_PREFIX",
                        skeleton_token[:SKELETON_PREFIX_LENGTH]
                    )
                )

    return variants


# ============================================================
# BLOCKING KEY GENERATION
# ============================================================

def create_block_keys(country, business_name, business_address):
    """
    Generate multiple blocking keys for a record.

    Existing Unicode strategies:

        NAME_PREFIX
        NAME_TOKEN
        NAME_TOKEN_PREFIX
        ADDRESS_TOKEN
        ADDRESS_TOKEN_PREFIX

    Additional multilingual name strategies:

        NAME_TRANSLIT
        NAME_TRANSLIT_PREFIX
        NAME_SKELETON
        NAME_SKELETON_PREFIX

    The transliteration/skeleton strategies are intended only
    to bridge different writing systems.

    Public function signature is unchanged.
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
    # 1. ORIGINAL UNICODE NAME PREFIX
    # ========================================================

    if normalized_name:

        prefix = normalized_name[:2]

        if prefix:

            keys.add(
                f"NAME_PREFIX|{country}|{prefix}"
            )

    # ========================================================
    # 2. ORIGINAL UNICODE NAME TOKEN BLOCKING
    # ========================================================

    name_tokens = tokenize_name(
        normalized_name
    )

    useful_name_tokens = _select_name_tokens(
        name_tokens
    )

    for token in useful_name_tokens:

        # --------------------------------------------
        # Exact Unicode token
        # --------------------------------------------

        keys.add(
            f"NAME_TOKEN|{country}|{token}"
        )

        # --------------------------------------------
        # Unicode token prefix
        # --------------------------------------------

        if len(token) >= TOKEN_PREFIX_LENGTH:

            keys.add(
                f"NAME_TOKEN_PREFIX|{country}|"
                f"{token[:TOKEN_PREFIX_LENGTH]}"
            )

        # ====================================================
        # MULTILINGUAL / TRANSLITERATION BLOCKING
        # ====================================================

        translit_variants = _transliteration_variants(
            token
        )

        for variant_type, variant in translit_variants:

            if variant_type == "FULL":

                keys.add(
                    f"NAME_TRANSLIT|{country}|{variant}"
                )

            elif variant_type == "PREFIX":

                keys.add(
                    f"NAME_TRANSLIT_PREFIX|{country}|{variant}"
                )

            elif variant_type == "SKELETON":

                keys.add(
                    f"NAME_SKELETON|{country}|{variant}"
                )

            elif variant_type == "SKELETON_PREFIX":

                keys.add(
                    f"NAME_SKELETON_PREFIX|{country}|{variant}"
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

        # Exact address token.
        keys.add(
            f"ADDRESS_TOKEN|{country}|{token}"
        )

        # Address token prefix.
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
    Build the multi-key blocking index.

    The DataFrame index is preserved as the candidate ID.

    Public interface is unchanged.
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

    Public interface is unchanged.
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

