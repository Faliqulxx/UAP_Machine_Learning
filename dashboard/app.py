import os
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow import keras
import gdown

# =============================
# BASE DIR (AMAN DI CLOUD)
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================
# CONFIG
# =============================
st.set_page_config(
    page_title="Klasifikasi Ras pada Kucing",
    page_icon="🐱",
    layout="centered"
)

# =============================
# LOAD LABELS
# =============================
@st.cache_data
def load_labels():
    label_path = os.path.join(BASE_DIR, "labels.txt")
    with open(label_path) as f:
        return [line.strip() for line in f.readlines()]

CLASS_NAMES = load_labels()

# =============================
# MODEL CONFIG
# =============================
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODELS = {
    "CNN Scratch": {
        "filename": "cnn_scratch_cat_breed_final.keras",
        "gdrive_id": "1EePG5jNTU6w5rJdZdIcVwfEjekzPcBwV"
    },
    "MobileNetV2": {
        "filename": "mobilenetv2_cat_breed_final_215.h5",
        "gdrive_id": "1O2oQdHw6EF1QpAg-o8zQAOl6KoQPLAJn"
    },
    "ResNet50": {
        "filename": "resnet50_cat_breed_final.keras",
        "gdrive_id": "1hxy8iW3R0mUWhUnU-uNeEQgd42UkGOAG"
    }
}

# =============================
# DOWNLOAD MODEL
# =============================
def download_model(model_info):
    model_path = os.path.join(MODEL_DIR, model_info["filename"])
    if not os.path.exists(model_path):
        with st.spinner(f"Mengunduh model {model_info['filename']}..."):
            gdown.download(
                f"https://drive.google.com/uc?id={model_info['gdrive_id']}",
                model_path,
                quiet=False
            )
    return model_path

# =============================
# LOAD MODEL (CACHE)
# =============================
@st.cache_resource
def load_model_cached(model_path):
    return keras.models.load_model(
        model_path,
        compile=False
    )

# =============================
# IMAGE PREPROCESS
# =============================
def preprocess_image(image, img_size=224):
    image = image.convert("RGB")
    image = image.resize((img_size, img_size))
    image = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(image, axis=0)

# =============================
# SIDEBAR
# =============================
st.sidebar.title("⚙️ Pengaturan Model")

model_name = st.sidebar.radio(
    "Pilih Model:",
    list(MODELS.keys())
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Performa Model (Test Set)")
st.sidebar.info(
    """
    **Akurasi Klasifikasi:**
    - CNN Scratch : **42%**
    - MobileNetV2 : **82% ⭐**
    - ResNet50    : **21%**
    """
)

# =============================
# LOAD MODEL
# =============================
model_info = MODELS[model_name]
model_path = download_model(model_info)
model = load_model_cached(model_path)

# =============================
# MAIN UI
# =============================
st.title("🐱 Klasifikasi Ras pada Kucing")
st.markdown(
    "Sistem klasifikasi **12 ras kucing** menggunakan "
    "**CNN Scratch & Transfer Learning**."
)

# =============================
# GALLERY
# =============================
st.subheader("📸 Contoh Ras Kucing")

cols = st.columns(4)
for idx, breed in enumerate(CLASS_NAMES):
    img_path = os.path.join(BASE_DIR, "sample_images", f"{breed}.jpg")
    if os.path.exists(img_path):
        with cols[idx % 4]:
            st.image(img_path, caption=breed.replace("_", " ").title(), width=140)

# =============================
# PREDICTION
# =============================
st.markdown("---")
st.subheader("🔍 Uji Model")

uploaded_file = st.file_uploader(
    "Upload gambar kucing",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar Input", width=350)

    if st.button("🔍 Prediksi"):
        with st.spinner("Melakukan prediksi..."):
            input_tensor = preprocess_image(image)
            preds = model.predict(input_tensor)[0]

            pred_index = np.argmax(preds)
            confidence = preds[pred_index]

        st.success("Prediksi berhasil!")

        st.markdown(f"""
        ### 🐾 Hasil Prediksi
        - **Model** : `{model_name}`
        - **Ras Kucing** : **{CLASS_NAMES[pred_index]}**
        - **Confidence** : **{confidence:.2%}**
        """)

        # =============================
        # PROBABILITY TABLE
        # =============================
        prob_df = pd.DataFrame({
            "Ras Kucing": CLASS_NAMES,
            "Probabilitas": preds
        }).sort_values("Probabilitas", ascending=False)

        st.subheader("📊 Probabilitas Semua Kelas")
        st.dataframe(prob_df, use_container_width=True)

# =============================
# FOOTER
# =============================
st.markdown("---")
st.caption("Faliqul Ishbah | Cat Breed Classification")
