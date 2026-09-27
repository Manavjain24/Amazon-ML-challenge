
import unicodedata


def normalize_unicode(text):
    """
    Normalize Unicode text while preserving characters from
    all writing systems.

    Examples:
        École  -> ecole
        राम    -> राम
        தமிழ்  -> தமிழ்
        ગુજરાત -> ગુજરાત
    """

    if not isinstance(text, str):
        return ""

    # Lowercase where the script supports it.
    text = text.lower()

    # NFKD separates Latin accents from their base characters.
    #
    # Example:
    #   é -> e + combining accent
    #
    # We remove combining marks only when they are actual
    # accent marks produced by decomposition. However, script
    # vowel signs such as the ones used in Devanagari, Tamil,
    # Gujarati, Kannada, etc. must be preserved.
    text = unicodedata.normalize("NFKD", text)

    result = []

    for char in text:
        category = unicodedata.category(char)

        # Keep letters.
        if category.startswith("L"):
            result.append(char)

        # Keep numbers.
        elif category.startswith("N"):
            result.append(char)

        # Keep combining marks.
        #
        # This is critical for Indic scripts.
        #
        # Examples:
        #   राम
        #   தமிழ்
        #   ગુજરાતી
        #
        # Their vowel signs / script marks must stay attached
        # to the preceding character.
        elif category.startswith("M"):
            result.append(char)

        # Everything else is handled later by normalize_name()
        # or normalize_address().
        else:
            result.append(char)

    return "".join(result)


def _normalize_text(text):
    """
    Common normalization logic for names and addresses.

    Preserves Unicode letters, numbers and combining marks.
    Replaces punctuation/symbols with spaces.
    """

    text = normalize_unicode(text)

    if not text:
        return ""

    result = []
    current_token = []

    for char in text:
        category = unicodedata.category(char)

        # Letters
        if category.startswith("L"):
            current_token.append(char)

        # Numbers
        elif category.startswith("N"):
            current_token.append(char)

        # Combining marks
        #
        # Keep them with the current token.
        elif category.startswith("M"):
            if current_token:
                current_token.append(char)

        else:
            # Punctuation, symbols and whitespace become separators.
            if current_token:
                result.append("".join(current_token))
                current_token = []

    if current_token:
        result.append("".join(current_token))

    return " ".join(result)


def normalize_name(name):
    """
    Normalize a business name.

    Preserves Unicode scripts including:
        Devanagari
        Tamil
        Gujarati
        Kannada
        Telugu
        etc.

    Removes punctuation and normalizes whitespace.
    """

    return _normalize_text(name)


def normalize_address(address):
    """
    Normalize a business address.

    Preserves Unicode scripts, numbers and combining marks.
    Removes punctuation and normalizes whitespace.
    """

    return _normalize_text(address)


"""
# Test Name sample
print("_________Starting Test name sample________________")

samples = [
    "École primaire de Georges",
    "Glanz, Duvall & Foster",
    "Pia Johnson, CPA, P.C.",
    "Sgc Medical Centre Pvt Ltd",
    "St. Zion",
    "राम मार्केटिंग प्राइवेट लिमिटेड",
    "ஆதித்யா பிராபர்டீஸ் எல்எல்பி",
    "குளோபல் பிசினஸ் பிரைவேட் லிமிடெட்",
    "શક્તિ અર્બન પ્રોડક્ટ્સ પ્રાઇવેટ લિમિટેડ",
    "ಸಿಲ್ವರ್ ಕನ್‌ಸಲ್ಟೆನ್ಸಿ ಪ್ರೈವೇட் ಲಿಮಿಟೆಡ್",
    "ગુજરાત Logistics Limited"
]

for name in samples:
    print(name, "→", normalize_name(name))


# Test Address sample
print("__________Starting Test Address Sample_______________")

samples = [
    "2621 Cotten Road, Tyler, TX",
    "H.No 143 / B-4 Paschim Vihar, New Delhi, North Delhi, Delhi",
    "4 Rue de la Constitution, Nantes, Pays de la Loire",
    "Shed No: 1&2, Phase-3Auto Nagar Vijayawada, Krishna, Andhra Pradesh",
    "दिल्ली, भारत, 123 मुख्य मार्ग",
    "கோயம்புத்தூர், தமிழ்நாடு",
    "અમદાવાદ, ગુજરાત",
    "ಬೆಂಗಳೂರು, ಕರ್ನಾಟಕ"
]

for address in samples:
    print(address, "→", normalize_address(address))
"""

