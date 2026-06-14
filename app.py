"""
Aplikasi Klasifikasi Buah dan Sayur
Mata Kuliah Pengolahan Citra Digital
Alif Lohen — 1304212117
"""

import streamlit as st
import numpy as np
import cv2
import json
import joblib
import os
import io
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Klasifikasi Buah & Sayur",
    page_icon="🍎",
    layout="wide"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title  { text-align:center; font-size:2.2rem; font-weight:700; color:#2d6a4f; }
    .sub-title   { text-align:center; font-size:1rem; color:#666; margin-bottom:1.5rem; }
    .metric-box  { background:#f0faf4; border:1px solid #b7e4c7; border-radius:10px;
                   padding:1rem; text-align:center; margin:4px; }
    .metric-val  { font-size:1.8rem; font-weight:700; color:#1b4332; }
    .metric-lbl  { font-size:0.85rem; color:#666; }
    .pred-box    { background:#e8f5e9; border-left:5px solid #2d6a4f; border-radius:8px;
                   padding:1rem 1.5rem; margin-top:1rem; }
    .pred-label  { font-size:1.6rem; font-weight:700; color:#1b4332; text-transform:capitalize; }
    .conf-text   { font-size:1rem; color:#444; }
    .wrong-box   { background:#fff3f3; border-left:5px solid #e63946; border-radius:8px;
                   padding:1rem 1.5rem; margin-top:1rem; }
</style>
""", unsafe_allow_html=True)

IMG_SIZE = (224, 224)

# ── Load artefak ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    artifacts = {}

    # Nama kelas
    if os.path.exists("class_names.json"):
        with open("class_names.json") as f:
            artifacts["classes"] = json.load(f)
    else:
        artifacts["classes"] = ["(class_names.json tidak ditemukan)"]

    # Model CNN
    try:
        import tensorflow as tf
        artifacts["cnn"] = tf.keras.models.load_model("mobilenetv3_buah_sayur.keras")
    except Exception as e:
        artifacts["cnn"] = None
        artifacts["cnn_error"] = str(e)

    # Model SVM
    try:
        artifacts["svm"]     = joblib.load("svm_model.pkl")
        artifacts["scaler"]  = joblib.load("scaler.pkl")
        artifacts["le"]      = joblib.load("label_encoder.pkl")
    except Exception as e:
        artifacts["svm"]     = None
        artifacts["svm_error"] = str(e)

    return artifacts

arts = load_artifacts()

# ── Fungsi prapemrosesan ──────────────────────────────────────────────────────
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
        lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab  = cv2.merge([clahe.apply(l), a, b])
        img  = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
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

# ── Fitur klasik untuk SVM ────────────────────────────────────────────────────
def extract_classic_features(img_rgb_float):
    from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
    img_uint = (img_rgb_float * 255).astype(np.uint8)
    gray     = cv2.cvtColor(img_uint, cv2.COLOR_RGB2GRAY)

    glcm  = graycomatrix(gray, distances=[1],
                         angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                         levels=256, symmetric=True, normed=True)
    feats = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
        feats.extend(graycoprops(glcm, prop).flatten().tolist())

    lbp   = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_h, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
    feats.extend(lbp_h.tolist())

    hsv   = cv2.cvtColor(img_uint, cv2.COLOR_RGB2HSV)
    h_h   = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    s_h   = cv2.calcHist([hsv], [1], None, [8],  [0, 256]).flatten()
    v_h   = cv2.calcHist([hsv], [2], None, [8],  [0, 256]).flatten()
    feats.extend((h_h / h_h.sum()).tolist())
    feats.extend((s_h / s_h.sum()).tolist())
    feats.extend((v_h / v_h.sum()).tolist())

    return np.array(feats, dtype=np.float32)

# ── Grad-CAM ──────────────────────────────────────────────────────────────────
def compute_gradcam(model, img_array, pred_index=None):
    import tensorflow as tf

    last_conv = [l.name for l in model.layers
                 if isinstance(l, (tf.keras.layers.Conv2D,
                                   tf.keras.layers.DepthwiseConv2D))]
    if not last_conv:
        return None
    last_conv_name = last_conv[-1]

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads    = tape.gradient(class_channel, conv_out)
    pooled   = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap  = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap  = tf.squeeze(heatmap)
    heatmap  = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap  = heatmap.numpy()

    heatmap_r = cv2.resize(heatmap, IMG_SIZE)
    colored   = cv2.applyColorMap(np.uint8(255 * heatmap_r), cv2.COLORMAP_JET)
    colored   = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return colored

# ═════════════════════════════════════════════════════════════════════════════
# UI LAYOUT
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-title">🍎 Klasifikasi Buah & Sayur</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Pengolahan Citra Digital · MobileNetV3 + Teknik Prapemrosesan</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Alif Lohen — 1304212117</div>', unsafe_allow_html=True)
st.divider()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pengaturan")

    st.subheader("🤖 Pilih Model")
    model_choice = st.radio(
        "Model klasifikasi:",
        ["MobileNetV3 (CNN)", "SVM + Fitur Klasik (Baseline)"],
        index=0
    )

    st.subheader("🔧 Prapemrosesan Citra")
    use_gaussian = st.toggle("Gaussian Filter",       value=True)
    use_median   = st.toggle("Median Filter",         value=True)
    use_clahe    = st.toggle("CLAHE",                 value=True)
    use_contrast = st.toggle("Contrast Stretching",   value=True)

    st.divider()
    st.subheader("📋 Kelas yang Tersedia")
    if arts["classes"]:
        for c in arts["classes"]:
            st.markdown(f"• {c.replace('_', ' ').title()}")

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📸 Klasifikasi", "📊 Tentang Model"])

with tab1:
    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.subheader("📤 Upload Gambar")
        uploaded = st.file_uploader(
            "Pilih gambar buah atau sayur",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        if uploaded:
            file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
            img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if img_bgr is None:
                st.error("Gagal membaca gambar. Coba file lain.")
                st.stop()

            # Prapemrosesan
            img_processed, stages = preprocess_image(
                img_bgr, use_gaussian, use_median, use_clahe, use_contrast
            )

            # Tampilkan tahap prapemrosesan
            st.markdown("**Tahap Prapemrosesan:**")
            stage_keys = list(stages.keys())
            cols = st.columns(len(stage_keys))
            for col, key in zip(cols, stage_keys):
                with col:
                    st.image(stages[key], caption=key, use_container_width=True)

    with col_result:
        if uploaded:
            st.subheader("🔍 Hasil Prediksi")

            # ── Prediksi CNN ──────────────────────────────────────────────────
            if "CNN" in model_choice:
                if arts["cnn"] is None:
                    st.error(f"Model CNN tidak ditemukan.\n{arts.get('cnn_error','')}")
                    st.info("Jalankan notebook training terlebih dahulu dan letakkan `mobilenetv3_buah_sayur.keras` di folder yang sama dengan `app.py`.")
                else:
                    img_input = np.expand_dims(img_processed, axis=0)
                    pred_prob = arts["cnn"].predict(img_input, verbose=0)[0]
                    pred_idx  = int(np.argmax(pred_prob))
                    pred_cls  = arts["classes"][pred_idx]
                    confidence = float(pred_prob[pred_idx])

                    st.markdown(f"""
                    <div class="pred-box">
                        <div class="pred-label">🎯 {pred_cls.replace('_',' ').title()}</div>
                        <div class="conf-text">Confidence: <b>{confidence:.1%}</b></div>
                        <div class="conf-text">Model: MobileNetV3</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Progress bar top-5
                    st.markdown("**Top-5 Prediksi:**")
                    top5_idx = np.argsort(pred_prob)[::-1][:5]
                    for idx in top5_idx:
                        label = arts["classes"][idx].replace('_', ' ').title()
                        prob  = float(pred_prob[idx])
                        st.progress(prob, text=f"{label}: {prob:.1%}")

                    # Grad-CAM
                    st.markdown("**Grad-CAM Visualization:**")
                    with st.spinner("Menghitung Grad-CAM..."):
                        gradcam = compute_gradcam(arts["cnn"], img_input, pred_idx)
                    if gradcam is not None:
                        orig_rgb = (img_processed * 255).astype(np.uint8)
                        overlay  = (gradcam * 0.45 + orig_rgb * 0.55).astype(np.uint8)
                        g1, g2   = st.columns(2)
                        g1.image(orig_rgb, caption="Citra Asli", use_container_width=True)
                        g2.image(overlay,  caption="Grad-CAM",   use_container_width=True)

            # ── Prediksi SVM ──────────────────────────────────────────────────
            else:
                if arts["svm"] is None:
                    st.error(f"Model SVM tidak ditemukan.\n{arts.get('svm_error','')}")
                    st.info("Jalankan notebook training dan letakkan `svm_model.pkl`, `scaler.pkl`, `label_encoder.pkl` di folder yang sama.")
                else:
                    with st.spinner("Mengekstraksi fitur klasik..."):
                        feats   = extract_classic_features(img_processed)
                        feats_s = arts["scaler"].transform([feats])
                        pred_enc = arts["svm"].predict(feats_s)[0]
                        pred_cls = arts["le"].inverse_transform([pred_enc])[0]
                        prob_arr = arts["svm"].predict_proba(feats_s)[0]
                        confidence = float(prob_arr.max())

                    st.markdown(f"""
                    <div class="pred-box">
                        <div class="pred-label">🎯 {pred_cls.replace('_',' ').title()}</div>
                        <div class="conf-text">Confidence: <b>{confidence:.1%}</b></div>
                        <div class="conf-text">Model: SVM + GLCM/LBP/HSV</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("**Top-5 Prediksi:**")
                    top5 = np.argsort(prob_arr)[::-1][:5]
                    for idx in top5:
                        label = arts["le"].inverse_transform([idx])[0].replace('_', ' ').title()
                        prob  = float(prob_arr[idx])
                        st.progress(prob, text=f"{label}: {prob:.1%}")

        else:
            st.info("⬅️ Upload gambar buah atau sayur untuk memulai klasifikasi.")

with tab2:
    st.subheader("📊 Tentang Model & Pipeline")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **Pipeline Prapemrosesan:**
        1. **Gaussian Filter** — reduksi noise Gaussian (kernel 3×3, σ=1)
        2. **Median Filter** — reduksi noise salt-and-pepper
        3. **CLAHE** — peningkatan kontras adaptif lokal (clipLimit=2, tile 8×8)
        4. **Contrast Stretching** — normalisasi min-max per channel
        5. **Resize & Normalisasi** — 224×224 px, nilai piksel [0,1]
        """)

    with c2:
        st.markdown("""
        **Ekstraksi Fitur & Klasifikasi:**
        - **Baseline:** GLCM (contrast, homogeneity, energy, correlation) + LBP + Histogram HSV → SVM RBF
        - **SOTA:** MobileNetV3Large pretrained ImageNet → Transfer Learning → Fine-tuning
        - **Evaluasi:** Accuracy, Precision, Recall, F1-Macro, Confusion Matrix, Grad-CAM
        """)

    st.divider()
    st.markdown("""
    **Dataset:** [Fruits and Vegetables Image Recognition](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition) — Kaggle  
    **Referensi:** Gonzalez & Woods (2018) · Howard et al. (2019) · He et al. (2016) · Ojala et al. (2002)  
    **Framework:** TensorFlow/Keras · OpenCV · scikit-learn · scikit-image · Streamlit
    """)
