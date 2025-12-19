import streamlit as st
import tensorflow as tf
import numpy as np
from utils.preprocess import preprocess_image

# =============================
# LOAD MODEL (CACHE)
# =============================
@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path, compile=False)

model_1 = load_model("models/mobilenetv2_cat_breed.keras")
model_2 = load_model("models/mobilenetv2_cat_breed_final.keras")

# Load class names
with open("class_names.txt") as f:
    class_names = [line.strip() for line in f.readlines()]

# =============================
# UI
# =============================
st.set_page_config(page_title="Klasifikasi Kucing", layout="centered")
st.title("🐱 Klasifikasi Ras Kucing")
st.markdown("Upload gambar kucing dan pilih model")

model_choice = st.selectbox(
    "Pilih Model",
    ("MobileNetV2 (Checkpoint)", "MobileNetV2 (Final)")
)

uploaded_file = st.file_uploader(
    "Upload Gambar",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    st.image(uploaded_file, caption="Gambar Input", width=300)
    img = preprocess_image(uploaded_file)

    if st.button("🔍 Prediksi"):
        model = model_1 if model_choice == "MobileNetV2 (Checkpoint)" else model_2
        prediction = model.predict(img)

        class_index = np.argmax(prediction)
        confidence = np.max(prediction)

        st.success(f"Prediksi: **{class_names[class_index]}**")
        st.write(f"Akurasi: **{confidence*100:.2f}%**")

        st.subheader("Probabilitas Tiap Kelas")
        st.bar_chart(prediction[0])
