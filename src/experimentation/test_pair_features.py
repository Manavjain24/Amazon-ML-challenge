from src.features.pair_features import extract_pair_features


source1_record = {
    "business_name": "Om Constructions Private Limited",
    "business_address": "Kolkata, West Bengal, India",
    "country": "India"
}


source2_record = {
    "business_name": "Om Constructions Pvt Ltd",
    "business_address": "Kolkata, West Bengal, India",
    "country": "India"
}


features = extract_pair_features(
    source1_record,
    source2_record
)


print("________ PAIR FEATURE TEST ________")

print("Source 1:")
print("Name:", source1_record["business_name"])
print("Address:", source1_record["business_address"])
print("Country:", source1_record["country"])

print("\nSource 2:")
print("Name:", source2_record["business_name"])
print("Address:", source2_record["business_address"])
print("Country:", source2_record["country"])

print("\nExtracted Features:")

for feature, value in features.items():
    print(f"{feature}: {value}")