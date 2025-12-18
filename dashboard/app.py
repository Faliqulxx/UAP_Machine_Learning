import os
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras

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
    with open("labels.txt") as f:
        return [line.strip() for line in f.readlines()]

CLASS_NAMES = load_labels()

# =============================
# LOAD MODELS
# =============================
@st.cache_resource
def load_model(model_path):
    return keras.models.load_model(model_path)

MODELS = {
    "CNN Scratch": load_model("models/cnn_scratch_cat_breed_final.keras"),
    "MobileNetV2": load_model("models/mobilenetv2_cat_breed_final.keras"),
    "ResNet50": load_model("models/resnet50_cat_breed_final.keras"),
}

# =============================
# IMAGE PREPROCESS
# =============================
def preprocess_image(image, img_size=224):
    image = image.convert("RGB")
    image = image.resize((img_size, img_size))
    image = np.array(image) / 255.0
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
    - MobileNetV2 : **82%** ⭐
    - ResNet50    : **21%**

    _Model terbaik: **MobileNetV2**_
    """
)


model = MODELS[model_name]

# =============================
# MAIN UI
# =============================
st.title("🐱 Klasifikasi Ras pada Kucing")
st.markdown(
    "Sistem klasifikasi **12 ras kucing** menggunakan **Deep Learning** "
    "(CNN Scratch & Transfer Learning)."
)

# =============================
# GALLERY
# =============================
st.subheader("📸 Contoh Ras Kucing dalam Dataset")

cols = st.columns(4)
for idx, breed in enumerate(CLASS_NAMES):
    img_path = f"sample_images/{breed}.jpg"
    if os.path.exists(img_path):
        with cols[idx % 4]:
            st.image(
                img_path,
                caption=breed.replace("_", " ").title(),
                width=140
            )

# =============================
# PREDICTION SECTION
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
        st.subheader("📊 Probabilitas Semua Kelas")
        prob_dict = {
            CLASS_NAMES[i]: float(preds[i])
            for i in range(len(CLASS_NAMES))
        }
        st.dataframe(
            prob_dict,
            use_container_width=True
        )

# =============================
# FOOTER
# =============================
st.markdown("---")
st.caption("Faliqul Isback | Cat Breed Classification")
