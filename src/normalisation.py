import pandas as pd
import re
import unicodedata
from pathlib import Path

PUNCTUATION_RE = re.compile(r"[^\w\s]")
WHITESPACE_RE = re.compile(r"\s+")

LEGAL_SUFFIXES = {
    "private", "limited", "ltd", "pvt", "inc", "incorporated",
    "corp", "corporation", "co", "company", "llc", "llp", "plc",
}


def normalize_text(text) -> str:
    """Lowercase, unicode-normalize, standardize '&', strip punctuation/whitespace."""
    if pd.isna(text):
        return ""
    text = unicodedata.normalize("NFKC", str(text)).lower().replace("&", " and ")
    text = PUNCTUATION_RE.sub("", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def remove_legal_suffixes(text: str) -> str:
    """Strip trailing legal/business suffix tokens (e.g. 'ltd', 'inc')."""
    tokens = text.split()
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def create_sorted_tokens(text: str) -> str:
    """Sorted, de-duplicated token representation for order-insensitive matching."""
    return " ".join(sorted(set(text.split())))


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full normalization pipeline to one source; original columns preserved."""
    df = df.copy()

    df["name_norm"] = df["business_name"].map(normalize_text)
    df["name_core"] = df["name_norm"].map(remove_legal_suffixes)
    df["name_tokens"] = df["name_norm"].map(create_sorted_tokens)
    df["name_compact"] = df["name_norm"].str.replace(" ", "", regex=False)

    df["address_norm"] = df["business_address"].map(normalize_text)
    df["address_tokens"] = df["address_norm"].map(create_sorted_tokens)
    df["address_compact"] = df["address_norm"].str.replace(" ", "", regex=False)

    df["country_norm"] = df["country"].map(normalize_text)

    return df


def main():
    data_dir = Path("dataset/train")
    output_dir = Path("data/normalized")
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = {}
    for i in (1, 2, 3):
        df = pd.read_csv(data_dir / f"train_source{i}.tsv", sep="\t")
        print(f"Source {i}: {len(df):,} rows")
        sources[i] = normalize_dataframe(df)

    for i, df in sources.items():
        out_path = output_dir / f"source{i}_normalized.tsv"
        df.to_csv(out_path, sep="\t", index=False)
        print(f"Saved -> {out_path}")

    sample_cols = [
        "business_name", "name_norm", "name_core", "name_tokens", "name_compact",
        "business_address", "address_norm", "address_tokens",
    ]
    print("\nSample (Source 1):")
    print(sources[1][sample_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()