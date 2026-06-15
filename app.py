"""
Aplikasi Klasifikasi Buah dan Sayur
Menggunakan MobileNetV3 dan Teknik Prapemrosesan Citra
Mata Kuliah Pengolahan Citra Digital
Alif Lohen - 1304212117
"""

import streamlit as st
import numpy as np
import cv2
import json
import os
from PIL import Image

st.set_page_config(page_title="Klasifikasi Buah & Sayur", page_icon="🍎", layout="wide")

st.markdown("""
<style>
    .main-title { text-align:center; font-size:2.2rem; font-weight:700; color:#2d6a4f; }
    .sub-title  { text-align:center; font-size:1rem; color:#666; margin-bottom:0.3rem; }
    .pred-box   { background:#e8f5e9; border-left:5px solid #2d6a4f; border-radius:8px;
                  padding:1rem 1.5rem; margin-top:1rem; }
    .pred-label { font-size:1.6rem; font-weight:700; color:#1b4332; text-transform:capitalize; }
    .conf-text  { font-size:1rem; color:#444; }
</style>
""", unsafe_allow_html=True)

IMG_SIZE = (224, 224)

@st.cache_resource
def load_artifacts():
    art = {}
    if os.path.exists("class_names.json"):
        with open("class_names.json") as f:
            art["classes"] = json.load(f)
    else:
        art["classes"] = None
        art["classes_error"] = "class_names.json tidak ditemukan"

    try:
        import tensorflow as tf
        art["model"] = tf.keras.models.load_model("mobilenetv3_buah_sayur.keras")
    except Exception as e:
        art["model"] = None
        art["model_error"] = str(e)
    return art

arts = load_artifacts()

def preprocess_image(img_bgr, use_gaussian=True, use_median=True,
                     use_clahe=True, use_contrast=True):
    img = img_bgr.copy()
    stages = {"Original": cv2.cvtColor(cv2.resize(img, IMG_SIZE), cv2.COLOR_BGR2RGB)}

    if use_gaussian:
        img = cv2.GaussianBlur(img, (3, 3), sigmaX=1.0)
        stages["Gaussian Filter"] = cv2.cvtColor(cv2.resize(img, IMG_SIZE), cv2.COLOR_BGR2RGB)

    if use_median:
        img = cv2.medianBlur(img, 3)
        stages["Median Filter"] = cv2.cvtColor(cv2.resize(img, IMG_SIZE), cv2.COLOR_BGR2RGB)

    if use_clahe:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab = cv2.merge([clahe.apply(l), a, b])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        stages["CLAHE"] = cv2.cvtColor(cv2.resize(img, IMG_SIZE), cv2.COLOR_BGR2RGB)

    if use_contrast:
        tmp = img.astype(np.float32)
        for c in range(3):
            mn, mx = tmp[:, :, c].min(), tmp[:, :, c].max()
            if mx > mn:
                tmp[:, :, c] = (tmp[:, :, c] - mn) / (mx - mn) * 255.0
        img = tmp.astype(np.uint8)
        stages["Contrast Stretching"] = cv2.cvtColor(cv2.resize(img, IMG_SIZE), cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    stages["Final Output"] = (img * 255).astype(np.uint8)
    return img, stages

def compute_gradcam(model, img_array, pred_index=None):
    import tensorflow as tf
    img_var = tf.Variable(tf.cast(img_array, tf.float32), trainable=False)
    with tf.GradientTape() as tape:
        tape.watch(img_var)
        preds = model(img_var, training=False)
        if pred_index is None:
            pred_index = int(tf.argmax(preds[0]))
        score = preds[:, pred_index]
    grads = tape.gradient(score, img_var)
    heatmap = tf.reduce_mean(tf.abs(grads[0]), axis=-1)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()
    h = cv2.resize(heatmap, IMG_SIZE)
    colored = cv2.applyColorMap(np.uint8(255 * h), cv2.COLORMAP_JET)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

st.markdown('<div class="main-title">🍎 Klasifikasi Buah & Sayur</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Pengolahan Citra Digital · MobileNetV3 + Teknik Prapemrosesan</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Alif Lohen — 1304212117</div>', unsafe_allow_html=True)
st.divider()

with st.sidebar:
    st.header("⚙️ Pengaturan")
    st.subheader("🔧 Prapemrosesan Citra")
    use_gaussian = st.toggle("Gaussian Filter", value=True)
    use_median   = st.toggle("Median Filter", value=True)
    use_clahe    = st.toggle("CLAHE", value=True)
    use_contrast = st.toggle("Contrast Stretching", value=True)
    st.divider()
    st.subheader("📋 Kelas Tersedia")
    if arts["classes"]:
        st.caption(", ".join(c.replace('_',' ').title() for c in arts["classes"]))

tab1, tab2 = st.tabs(["📸 Klasifikasi", "📊 Tentang Model"])

with tab1:
    col_up, col_res = st.columns([1, 1], gap="large")

    with col_up:
        st.subheader("📤 Upload Gambar")
        uploaded = st.file_uploader("Pilih gambar buah/sayur",
                                    type=["jpg","jpeg","png"], label_visibility="collapsed")
        if uploaded:
            file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img_bgr is None:
                st.error("Gagal membaca gambar.")
                st.stop()

            img_processed, stages = preprocess_image(
                img_bgr, use_gaussian, use_median, use_clahe, use_contrast)

            st.markdown("**Tahap Prapemrosesan:**")
            keys = list(stages.keys())
            cols = st.columns(len(keys))
            for col, key in zip(cols, keys):
                col.image(stages[key], caption=key, use_container_width=True)

    with col_res:
        if uploaded:
            st.subheader("🔍 Hasil Prediksi")
            if arts["model"] is None:
                st.error("Model tidak ditemukan.")
                st.info(arts.get("model_error", ""))
                st.stop()
            if arts["classes"] is None:
                st.error("class_names.json tidak ditemukan.")
                st.stop()

            img_input = np.expand_dims(img_processed, axis=0)
            pred_prob = arts["model"].predict(img_input, verbose=0)[0]
            pred_idx  = int(np.argmax(pred_prob))
            pred_cls  = arts["classes"][pred_idx]
            conf      = float(pred_prob[pred_idx])

            st.markdown(f"""
            <div class="pred-box">
                <div class="pred-label">🎯 {pred_cls.replace('_',' ').title()}</div>
                <div class="conf-text">Confidence: <b>{conf:.1%}</b></div>
                <div class="conf-text">Model: MobileNetV3</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Top-5 Prediksi:**")
            top5 = np.argsort(pred_prob)[::-1][:5]
            for idx in top5:
                label = arts["classes"][idx].replace('_',' ').title()
                st.progress(float(pred_prob[idx]), text=f"{label}: {pred_prob[idx]:.1%}")

            st.markdown("**Grad-CAM:**")
            with st.spinner("Menghitung Grad-CAM..."):
                gradcam = compute_gradcam(arts["model"], img_input, pred_idx)
            orig = (img_processed * 255).astype(np.uint8)
            overlay = (gradcam * 0.45 + orig * 0.55).astype(np.uint8)
            g1, g2 = st.columns(2)
            g1.image(orig, caption="Citra (setelah preprocessing)", use_container_width=True)
            g2.image(overlay, caption="Grad-CAM", use_container_width=True)
        else:
            st.info("⬅️ Upload gambar untuk memulai klasifikasi.")

with tab2:
    st.subheader("📊 Tentang Model & Pipeline")
    st.markdown("""
    **Pipeline Prapemrosesan:**
    1. **Gaussian Filter** — reduksi noise Gaussian
    2. **Median Filter** — reduksi noise impulsif
    3. **CLAHE** — peningkatan kontras lokal
    4. **Contrast Stretching** — perbaikan distribusi intensitas global
    5. **Resize & Normalisasi** — 224×224, nilai piksel [0,1]

    **Klasifikasi:** MobileNetV3Large pretrained ImageNet → Transfer Learning → Fine-tuning
    **Ekstraksi fitur:** otomatis oleh lapisan konvolusional MobileNetV3
    **Evaluasi:** Accuracy, Precision, Recall, F1-Macro, Confusion Matrix, Grad-CAM

    **Dataset:** Fruit and Vegetable Image Recognition (Kaggle)
    **Framework:** TensorFlow/Keras · OpenCV · Streamlit
    """)
