import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import pandas as pd
import json
import os
import datetime
import folium
from streamlit_folium import st_folium
from keras.models import load_model

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="EcoScan AI — Smart Waste Detection",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ─────────────────────────────────────────────
#  MODEL LOADING & PREDICTION (From main.py)
# ─────────────────────────────────────────────
@st.cache_resource
def load_keras_model():
    # Load the model and labels provided in the workspace
    model = load_model("final_model.h5", compile=False)
    with open("final_labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return model, labels


model, class_names = load_keras_model()


def process_and_predict(image_data):
    # exact preprocessing steps from main.py
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = float(prediction[0][index])

    # Class name cleaning (e.g., "0 Dirty" -> "Dirty")
    display_label = class_name.split(' ', 1)[1] if ' ' in class_name else class_name
    return display_label, confidence_score


# ─────────────────────────────────────────────
#  SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────
if 'detections' not in st.session_state:
    st.session_state.detections = []
if 'total_scans' not in st.session_state:
    st.session_state.total_scans = 0
if 'clean_count' not in st.session_state:
    st.session_state.clean_count = 0
if 'dirty_count' not in st.session_state:
    st.session_state.dirty_count = 0

# ─────────────────────────────────────────────
#  DESIGN SYSTEM + GLOBAL CSS (From fixed.txt)
# ─────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg-base:       #04080f; --bg-surface:    #080f1c; --bg-card:       rgba(255,255,255,0.035);
  --border:        rgba(255,255,255,0.08); --border-accent: rgba(0,224,143,0.35);
  --text-primary:  #eef5f2; --text-secondary:#8fa8a0; --green:         #00e08f;
  --blue:          #0099ff; --red:           #ff4757; --amber:         #ffb800;
  --radius-md:     16px; --radius-lg:     24px;
}
html, body, [data-testid="stAppViewContainer"] { background: var(--bg-base) !important; color: var(--text-primary); font-family: 'Outfit', sans-serif; }
.card { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 24px; backdrop-filter: blur(18px); margin-bottom: 20px; }
.hero h1 { font-size: clamp(2rem, 6.5vw, 4rem); font-weight: 800; background: linear-gradient(140deg, #fff, var(--green), var(--blue)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.stat-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 20px; text-align: center; }
.stat-value { font-size: 2rem; font-weight: 800; }
.result-wrap { border-radius: var(--radius-md); padding: 20px; margin-top: 15px; }
.result-wrap.clean { background: rgba(0,224,143,0.1); border: 1px solid var(--green); }
.result-wrap.dirty { background: rgba(255,71,87,0.1); border: 1px solid var(--red); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MAIN UI LAYOUT
# ─────────────────────────────────────────────
st.markdown('<div class="app-shell">', unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero">
    <div style="color:var(--green); letter-spacing:0.1em; font-weight:600; margin-bottom:10px;">AI POWERED ENVIRONMENT PROTECTION</div>
    <h1>EcoScan AI</h1>
    <p style="color:var(--text-secondary); max-width:600px; margin:0 auto;">Detecting waste categories in real-time to promote a cleaner future.</p>
</div>
""", unsafe_allow_html=True)

# Stats Strip
acc = (st.session_state.clean_count / st.session_state.total_scans * 100) if st.session_state.total_scans > 0 else 0
c1, c2, c3, c4 = st.columns(4)
with c1: st.markdown(
    f'<div class="stat-card"><div class="stat-label">Total Scans</div><div class="stat-value">{st.session_state.total_scans}</div></div>',
    unsafe_allow_html=True)
with c2: st.markdown(
    f'<div class="stat-card"><div class="stat-label" style="color:var(--green)">Clean Items</div><div class="stat-value">{st.session_state.clean_count}</div></div>',
    unsafe_allow_html=True)
with c3: st.markdown(
    f'<div class="stat-card"><div class="stat-label" style="color:var(--red)">Dirty Items</div><div class="stat-value">{st.session_state.dirty_count}</div></div>',
    unsafe_allow_html=True)
with c4: st.markdown(
    f'<div class="stat-card"><div class="stat-label">Environment Score</div><div class="stat-value">{acc:.1f}%</div></div>',
    unsafe_allow_html=True)

tabs = st.tabs(["🔍 Environmental Scan", "📊 History & Analytics"])

# ── TAB 1: SCAN ──────────────────────────────
with tabs[0]:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="card"><h3>📸 Input Source</h3>', unsafe_allow_html=True)
        source = st.radio("Choose Input:", ["Camera Stream", "Upload File"], horizontal=True)

        # Latitude and Longitude for the map
        lat = st.number_input("Scan Latitude", value=40.7128, format="%.4f")
        lon = st.number_input("Scan Longitude", value=-74.0060, format="%.4f")

        img_file = None
        if source == "Camera Stream":
            img_file = st.camera_input("Capture live waste data")
        else:
            img_file = st.file_uploader("Select environmental image...", type=["jpg", "png", "jpeg"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card"><h3>🔍 Live Analysis</h3>', unsafe_allow_html=True)
        if img_file:
            image = Image.open(img_file).convert("RGB")
            label, confidence = process_and_predict(image)

            # Logic to update states
            if st.button("Log Detection"):
                st.session_state.total_scans += 1
                if "Clean" in label:
                    st.session_state.clean_count += 1
                else:
                    st.session_state.dirty_count += 1

                st.session_state.detections.append({
                    "time": datetime.datetime.now().strftime("%H:%M:%S"),
                    "result": label,
                    "conf": f"{confidence:.2%}",
                    "lat": lat,
                    "lon": lon
                })
                st.success("Detection logged to history!")

            # Display Results
            res_class = "clean" if "Clean" in label else "dirty"
            emoji = "✅" if "Clean" in label else "⚠️"

            st.markdown(f"""
                <div class="result-wrap {res_class}">
                    <div style="font-size:2rem;">{emoji} {label}</div>
                    <div style="opacity:0.8;">Confidence: {confidence:.2%}</div>
                </div>
            """, unsafe_allow_html=True)
            st.image(image, use_container_width=True)
        else:
            st.markdown("""
                <div style="text-align:center; padding:50px; opacity:0.3;">
                    <div style="font-size:3rem;">📷</div>
                    <p>Waiting for image input...</p>
                </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ── TAB 2: HISTORY ────────────────────────────
with tabs[1]:
    m_col, h_col = st.columns([2, 1])

    with m_col:
        st.markdown('<div class="card"><h3>🌍 Geospatial Distribution</h3>', unsafe_allow_html=True)
        # Center map on last detection or default
        center_lat = st.session_state.detections[-1]['lat'] if st.session_state.detections else 40.7128
        center_lon = st.session_state.detections[-1]['lon'] if st.session_state.detections else -74.0060

        m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB dark_matter")
        for d in st.session_state.detections:
            color = "green" if "Clean" in d['result'] else "red"
            folium.CircleMarker(
                location=[d['lat'], d['lon']],
                radius=8, color=color, fill=True,
                popup=f"{d['result']} ({d['conf']})"
            ).add_to(m)
        st_folium(m, width="100%", height=400)
        st.markdown('</div>', unsafe_allow_html=True)

    with h_col:
        st.markdown('<div class="card"><h3>📋 History Logs</h3>', unsafe_allow_html=True)
        if st.session_state.detections:
            df = pd.DataFrame(st.session_state.detections)
            st.dataframe(df[['time', 'result', 'conf']], use_container_width=True, hide_index=True)
        else:
            st.info("No detections recorded yet.")
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # End App Shell