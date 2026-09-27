import pandas as pd

from src.preprocessing.normalization import (
    normalize_name,
    normalize_address
)

INPUT_FILE = "data/dataset/test/test_source1.tsv"
TEMP_FILE = "data/samples/normalized_sample.tsv"

# Read only — original file is never modified
df = pd.read_csv(
    INPUT_FILE,
    sep="\t",
    nrows=1000
)

# Create normalized columns in memory
df["normalized_business_name"] = df["business_name"].apply(normalize_name)
df["normalized_business_address"] = df["business_address"].apply(
    normalize_address
)

# Write ONLY the small test sample
df.to_csv(
    TEMP_FILE,
    sep="\t",
    index=False
)

print(f"Saved normalized test sample to {TEMP_FILE}")
print("_____ NON ASCII Chars testing__________")
samples = [
    "राम मार्केटिंग प्राइवेट लिमिटेड",
    "आदित्य प्रॉपर्टीज एलएलपी",
    "குளோபல் பிசினஸ் பிரைவேட் லிமிடெட்",
    "શક્તિ અર્બન પ્રોડક્ટ્સ પ્રાઇવેટ લિમિટેડ",
    "ಸಿಲ್ವರ್ ಕನ್‌ಸಲ್ಟೆನ್ಸಿ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್",
    "ગુજરાત Logistics Limited",
]

for name in samples:
    print(name, "→", normalize_name(name))