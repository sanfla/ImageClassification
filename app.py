import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from huggingface_hub import hf_hub_download

st.title("Klasifikasi Gambar: Kucing vs Anjing")

model_path = hf_hub_download(
    repo_id="sanfla/models_CatDog", 
    filename="cnnModel.h5"
)

model = tf.keras.models.load_model(model_path, compile=False)

def preprocess(img):
    img = img.resize((128, 128))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

uploaded = st.file_uploader("Upload gambar", type=['jpg', 'png', 'jpeg'])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Gambar diunggah", use_column_width=True)
    input_img = preprocess(img)

    pred = model.predict(input_img)[0][0]
    label = "Kucing" if pred > 0.5 else "Anjing"
    st.subheader(f"Prediksi: {label}")







