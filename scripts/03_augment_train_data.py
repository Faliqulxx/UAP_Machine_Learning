import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, save_img

# =========================
# PATH
# =========================
TRAIN_DIR = "../processed_dataset/train"
AUG_PER_IMAGE = 3   # jumlah augmentasi per gambar

# =========================
# IMAGE AUGMENTATION SETUP
# =========================
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode="nearest"
)

# =========================
# AUGMENTASI
# =========================
total_generated = 0

for breed in os.listdir(TRAIN_DIR):
    breed_path = os.path.join(TRAIN_DIR, breed)

    if not os.path.isdir(breed_path):
        continue

    images = os.listdir(breed_path)

    for img_name in images:
        img_path = os.path.join(breed_path, img_name)

        try:
            img = load_img(img_path, target_size=(224, 224))
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            prefix = img_name.split(".")[0]

            i = 0
            for batch in datagen.flow(
                x,
                batch_size=1,
                save_to_dir=breed_path,
                save_prefix=prefix + "_aug",
                save_format="jpg"
            ):
                i += 1
                total_generated += 1
                if i >= AUG_PER_IMAGE:
                    break

        except Exception as e:
            print(f"❌ Error pada {img_name}: {e}")

print(f"✅ Total gambar augmentasi dibuat: {total_generated}")