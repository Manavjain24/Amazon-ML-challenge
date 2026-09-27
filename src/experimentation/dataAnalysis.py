import pandas as pd
import re
input_file = "data/samples/normalized_sample.tsv"
df = pd.read_csv(
    input_file,
    sep="\t"
)

print(df.head())
print(df.columns)
print(df.shape)
# Trying to see uniqe and duplicate data
print("\nMissing values:")
print(df.isna().sum())

print("\nCountry distribution:")
print(df["country"].value_counts())

print("\nDuplicate business names:")
print(df["business_name"].duplicated().sum())

print("\nDuplicate addresses:")
print(df["business_address"].duplicated().sum())

print("\nUnique countries:")
print(df["country"].nunique())

print("\nLength Distribution for each field!! ")
print("\nBusiness name length:")
print(df["business_name"].str.len().describe())

print("\nBusiness address length:")
print(df["business_address"].str.len().describe())
print("\n puntuaation etc check _______")
print("\nSample business names:")
print(df["business_name"].sample(20, random_state=42).to_list())

print("\nSample business addresses:")
print(df["business_address"].sample(10, random_state=42).to_list())

print("_____ Trying tokenizationss for normalization strategy_______")
def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


print("\n_________ NAME TOKENS _________")

sample_names = df["business_name"].sample(20, random_state=42)

for name in sample_names:
    print(f"{name} → {tokenize(name)}")


print("\n_________ ADDRESS TOKENS _________")

sample_addresses = df["business_address"].sample(10, random_state=42)

for address in sample_addresses:
    print(f"{address} → {tokenize(address)}")