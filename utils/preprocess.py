import numpy as np
from PIL import Image

IMG_SIZE = 224

def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0  # SAMA dengan ImageDataGenerator
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
