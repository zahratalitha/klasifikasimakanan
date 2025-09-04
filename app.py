import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os


st.title("🍔🥗🍣 Klasifikasi Gambar Makanan")


MODEL_URL = "https://github.com/zahratalitha/food101/raw/main/cnn_food101_model%20(1).h5"
MODEL_PATH = "cnn_food101_model.h5"

if not os.path.exists(MODEL_PATH):
    st.write("📥 Mengunduh model dari GitHub...")
    r = requests.get(MODEL_URL)
    if r.status_code == 200:
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        st.success("✅ Model berhasil diunduh!")
    else:
        st.error("❌ Gagal mendownload model. Periksa URL raw GitHub.")

st.sidebar.header("📤 Upload Gambar")
uploaded_file = st.sidebar.file_uploader("Pilih gambar makanan", type=["jpg", "jpeg", "png"])


IMG_HEIGHT, IMG_WIDTH = 224,224
def preprocess_image_cnn(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalisasi 0-1
    return img_array

if uploaded_file is not None:
    st.image(uploaded_file, caption="Gambar yang diupload", use_column_width=True)
    
    img_array = preprocess_image_cnn(uploaded_file)
    preds = model.predict(img_array)
    
    # Ambil top-5 prediksi
    top5_idx = preds[0].argsort()[-5:][::-1]
    top5_probs = preds[0][top5_idx]
    
   
    with open("labels.txt", "r") as f:
        class_labels = [line.strip() for line in f.readlines()]
    top5_labels = [class_labels[i] for i in top5_idx]
    

    st.subheader("🍽 Prediksi Makanan Utama")
    st.write(f"**{top5_labels[0]}** dengan probabilitas {top5_probs[0]*100:.2f}%")
    
    st.subheader("📌 Top 5 Prediksi")
    for label, prob in zip(top5_labels, top5_probs):
        st.write(f"{label}: {prob*100:.2f}%")
