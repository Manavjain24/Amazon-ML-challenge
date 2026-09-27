import pandas as pd

from src.preprocessing.tokenizer import tokenize_name
from src.features.string_similarity import jaccard_similarity
from src.features.string_similarity import levenshtein_similarity

INPUT_FILE = "data/samples/normalized_sample.tsv"

df = pd.read_csv(
    INPUT_FILE,
    sep="\t"
)

print("________ NAME JACCARD TEST ________")

#name_a = df["business_name"].iloc[0]
#name_b = df["business_name"].iloc[1]
name_a = "Om Constructions Pvt Ltd"
name_b = "Om Constructions Private Limited"
tokens_a = tokenize_name(name_a)
tokens_b = tokenize_name(name_b)

score = jaccard_similarity(tokens_a, tokens_b)

print("Name A:", name_a)
print("Tokens A:", tokens_a)

print("Name B:", name_b)
print("Tokens B:", tokens_b)

print("Jaccard similarity:", score)


print("___________________- Levenshtein similarity______________")
pairs = [
    ("google", "google"),
    ("google", "gogle"),
    ("microsoft", "microsft"),
    ("google", "amazon"),
]


for a, b in pairs:
    score = levenshtein_similarity(a, b)

    print(f"{a} ↔ {b}")
    print(f"Similarity: {score:.3f}")
    print()