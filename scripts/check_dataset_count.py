import os

BASE_DIR = "../processed_dataset"

for split in ["train", "val", "test"]:
    total = 0
    for breed in os.listdir(os.path.join(BASE_DIR, split)):
        breed_dir = os.path.join(BASE_DIR, split, breed)
        total += len(os.listdir(breed_dir))
    print(f"{split.upper()} : {total} images")
