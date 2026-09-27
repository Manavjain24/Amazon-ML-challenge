import pandas as pd

s1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t"
)

s2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t"
)

s3 = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t"
)

gt = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)

print(s1.head())
print(s1.shape)
print(s2.shape)
print(s3.shape)
print(gt.head())
