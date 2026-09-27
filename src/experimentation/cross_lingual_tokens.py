from src.blocking.candidate_generation import create_block_keys

pairs = [
    ("India", "Ss Food Private Limited", "Af-684, Nandgram, Ghaziabad"),
    ("India", "एसएस फूड प्राइवेट लिमिटेड", "AF-0684, NANDGRAM, GHAZIABAD"),
    ("India", "Red Ventures Private Limited", ""),
    ("India", "रेड वेंचर्स प्राइवेट लिमिटेड", ""),
]

for country, name, address in pairs:
    keys = create_block_keys(country, name, address)

    print("\n", name)
    for key in sorted(keys):
        if "TRANSLIT" in key or "SKELETON" in key:
            print(key)