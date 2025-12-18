import os
import shutil

# =========================
# PATH
# =========================
RAW_DIR = "../raw_dataset"
IMAGES_DIR = os.path.join(RAW_DIR, "images")
ANNOT_FILE = os.path.join(RAW_DIR, "annotations", "list.txt")

OUTPUT_DIR = "../processed_dataset/all_cats"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# BACA ANNOTATION
# =========================
cat_count = 0

with open(ANNOT_FILE, "r") as f:
    for line in f:
        if line.startswith("#"):
            continue

        filename, class_id, species, breed_id = line.strip().split()

        # species == 1 → KUCING
        if int(species) == 1:
            breed = filename.rsplit("_", 1)[0].lower()
            src_img = os.path.join(IMAGES_DIR, filename + ".jpg")

            breed_dir = os.path.join(OUTPUT_DIR, breed)
            os.makedirs(breed_dir, exist_ok=True)

            dst_img = os.path.join(breed_dir, filename + ".jpg")
            shutil.copy(src_img, dst_img)
            cat_count += 1

print(f"✅ Total gambar kucing berhasil diekstrak: {cat_count}")
