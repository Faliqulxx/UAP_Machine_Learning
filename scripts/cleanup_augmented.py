# scripts/cleanup_augmented.py
import os

TRAIN_DIR = "../processed_dataset/train"

for breed in os.listdir(TRAIN_DIR):
    path = os.path.join(TRAIN_DIR, breed)
    for img in os.listdir(path):
        if "_aug" in img:
            os.remove(os.path.join(path, img))

print("✅ File augmentasi lama dihapus")
