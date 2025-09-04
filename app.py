import streamlit as st
import tensorflow as tf
import os
import requests
from PIL import Image
import numpy as np

# -----------------------------
# Judul Aplikasi
# -----------------------------
st.title("🍔🥗🍣 Food-101 Image Classification (CNN dari Nol)")
st.write("Upload gambar makanan untuk mendapatkan prediksi top-1 dan top-5 berdasarkan model CNN yang telah dilatih.")

# -----------------------------
# Download model dari GitHub
# -----------------------------
MODEL_URL = "https://github.com/zahratalitha/food101/raw/main/cnn_food101_model%20(1).h5"
MODEL_PATH = "cnn_food101_model.h5"

if not os.path.exists(MODEL_PATH):
    st.write("📥 Mengunduh model dari GitHub...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
    st.success("✅ Model berhasil diunduh!")

# -----------------------------
# Load model
# -----------------------------
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success("✅ Model berhasil dimuat!")
    st.write("Input shape model:", model.input_shape)
    st.write("Output shape model:", model.output_shape)
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    model = None

# -----------------------------
# Sidebar: Upload Gambar
# -----------------------------
st.sidebar.header("📤 Upload Gambar")
uploaded_file = st.sidebar.file_uploader("Pilih gambar makanan", type=["jpg", "jpeg", "png"])

# -----------------------------
# Preprocessing gambar
# -----------------------------
IMG_HEIGHT, IMG_WIDTH = 128, 128  # Sesuai training

def preprocess_image_cnn(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# -----------------------------
# Prediksi
# -----------------------------
if uploaded_file is not None:
    if model is not None:
        # Tampilkan gambar
        st.image(uploaded_file, caption="Gambar yang diupload", use_container_width=True)

        # Preprocess & prediksi
        img_array = preprocess_image_cnn(uploaded_file)
        preds = model.predict(img_array)

        # Top-5 prediksi
        top5_idx = preds[0].argsort()[-5:][::-1]
        top5_probs = preds[0][top5_idx]

        # Load labels
        if os.path.exists("labels.txt"):
            with open("labels.txt", "r") as f:
                class_labels = [line.strip() for line in f.readlines()]
            top5_labels = [class_labels[i] for i in top5_idx]
        else:
            top5_labels = [f"Kelas {i}" for i in top5_idx]

        # Tampilkan top-1
        st.subheader("🍽 Prediksi Makanan Utama")
        st.write(f"**{top5_labels[0]}** dengan probabilitas {top5_probs[0]*100:.2f}%")

        # Tampilkan top-5
        st.subheader("📌 Top 5 Prediksi")
        for label, prob in zip(top5_labels, top5_probs):
            st.write(f"{label}: {prob*100:.2f}%")
    else:
        st.error("❌ Model tidak tersedia. Tidak dapat melakukan prediksi.")
