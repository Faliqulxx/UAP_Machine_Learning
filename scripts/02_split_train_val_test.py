import os
import shutil
import random

SOURCE_DIR = "../processed_dataset/all_cats"
TARGET_DIR = "../processed_dataset"

SPLIT_RATIO = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

random.seed(42)

for split in SPLIT_RATIO:
    os.makedirs(os.path.join(TARGET_DIR, split), exist_ok=True)

for breed in os.listdir(SOURCE_DIR):
    breed_path = os.path.join(SOURCE_DIR, breed)
    images = os.listdir(breed_path)
    random.shuffle(images)

    total = len(images)
    train_end = int(total * SPLIT_RATIO["train"])
    val_end = train_end + int(total * SPLIT_RATIO["val"])

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, imgs in splits.items():
        split_breed_dir = os.path.join(TARGET_DIR, split, breed)
        os.makedirs(split_breed_dir, exist_ok=True)

        for img in imgs:
            shutil.copy(
                os.path.join(breed_path, img),
                os.path.join(split_breed_dir, img)
            )

print("✅ Split train / val / test selesai")
