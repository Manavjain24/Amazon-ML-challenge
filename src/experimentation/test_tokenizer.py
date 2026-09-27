import pandas as pd

from src.preprocessing.tokenizer import (
    tokenize_name,
    tokenize_address
)

INPUT_FILE = "data/samples/normalized_sample.tsv"

df = pd.read_csv(
    INPUT_FILE,
    sep="\t"
)

print("________ NAME TOKENS ________")

for name in df["business_name"].head(10):
    print(name, "→", tokenize_name(name))

print("\n________ ADDRESS TOKENS ________")

for address in df["business_address"].head(10):
    print(address, "→", tokenize_address(address))
    print("\n________ NON-ASCII NAME TOKENS ________")

non_ascii_names = [
    "राम मार्केटिंग प्राइवेट लिमिटेड",
    "ஆதித்யா பிராபர்டீஸ் எல்எல்பி",
    "குளோபல் பிசினஸ் பிரைவேட் லிமிடெட்",
    "શક્તિ અર્બન પ્રોડક્ટ્સ પ્રાઇવેટ લિમિટેડ",
    "ಸಿಲ್ವರ್ ಕನ್‌ಸಲ್ಟೆನ್ಸಿ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್",
    "ગુજરાત Logistics Limited",
]

for name in non_ascii_names:
    print(
        name,
        "→",
        tokenize_name(name)
    )
print("\n________ NON-ASCII ADDRESS TOKENS ________")

non_ascii_addresses = [
    "दिल्ली, भारत, 123 मुख्य मार्ग",
    "கோயம்புத்தூர், தமிழ்நாடு",
    "અમદાવાદ, ગુજરાત",
    "ಬೆಂಗಳೂರು, ಕರ್ನಾಟಕ",
]

for address in non_ascii_addresses:
    print(
        address,
        "→",
        tokenize_address(address)
    )   