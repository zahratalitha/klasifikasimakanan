# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os

# -----------------------------
# Judul
# -----------------------------
st.title("🍔🥗🍣 Food-101 Image Classification (CNN dari Nol)")


MODEL_URL = "https://github.com/zahratalitha/food101/blob/b8182f1ca13ae214921c6e125e5a20dafe2f9946/cnn_food101_model%20(1).h5"
MODEL_PATH = "cnn_food101_model (1).h5"

if not os.path.exists(MODEL_PATH):
    st.write("📥 Mengunduh model dari GitHub...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
    st.success("✅ Model berhasil diunduh!")

st.write("📥 Memuat model CNN...")
model = tf.keras.models.load_model(MODEL_PATH)
st.success("✅ Model CNN berhasil dimuat!")

st.write("Input shape model:", model.input_shape)
st.write("Output shape model:", model.output_shape)

# -----------------------------
# Sidebar: Upload Gambar
# -----------------------------
st.sidebar.header("📤 Upload Gambar")
uploaded_file = st.sidebar.file_uploader("Pilih gambar makanan", type=["jpg", "jpeg", "png"])

# -----------------------------
# Preprocessing CNN dari Nol
# -----------------------------
IMG_HEIGHT, IMG_WIDTH = 224, 224  # disesuaikan dengan model CNN

def preprocess_image_cnn(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalisasi 0-1
    return img_array

# -----------------------------
# Prediksi
# -----------------------------
if uploaded_file is not None:
    # Tampilkan gambar di halaman utama
    st.image(uploaded_file, caption="Gambar yang diupload", use_column_width=True)

    # Preprocess & prediksi
    img_array = preprocess_image_cnn(uploaded_file)
    preds = model.predict(img_array)

    # Ambil top-5 prediksi
    top5_idx = preds[0].argsort()[-5:][::-1]
    top5_probs = preds[0][top5_idx]

    # Gunakan file labels.txt jika tersedia
    try:
        with open("labels.txt", "r") as f:
            class_labels = [line.strip() for line in f.readlines()]
        top5_labels = [class_labels[i] for i in top5_idx]
    except:
        top5_labels = [f"Kelas {i}" for i in top5_idx]

    # Tampilkan hasil
    st.subheader("📌 Top 5 Prediksi")
    for label, prob in zip(top5_labels, top5_probs):
        st.write(f"{label}: {prob*100:.2f}%")
