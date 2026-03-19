from __future__ import annotations

from datetime import datetime
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from streamlit_folium import st_folium
from tensorflow.keras.models import load_model


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "final_model.h5"
LABELS_PATH = ROOT / "final_labels.txt"
IMAGE_SIZE = (224, 224)


st.set_page_config(
    page_title="EcoScan AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def load_assets():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Missing label file: {LABELS_PATH}")

    model = load_model(MODEL_PATH, compile=False)
    labels = [line.strip() for line in LABELS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    return model, labels


def clean_label(label: str) -> str:
    return label.split(" ", 1)[1] if " " in label else label


def predict_image(image: Image.Image) -> tuple[str, float]:
    model, labels = load_assets()

    prepared = ImageOps.fit(image.convert("RGB"), IMAGE_SIZE, Image.Resampling.LANCZOS)
    image_array = np.asarray(prepared)
    normalized = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized

    prediction = model.predict(data, verbose=0)
    best_index = int(np.argmax(prediction))
    return clean_label(labels[best_index]), float(prediction[0][best_index])


if "detections" not in st.session_state:
    st.session_state.detections = []


st.markdown(
    """
    <style>
    :root {
      --bg-base: #07111d;
      --panel: rgba(255, 255, 255, 0.05);
      --panel-border: rgba(255, 255, 255, 0.10);
      --text-muted: #98a9b9;
      --green: #2ed573;
      --red: #ff6b6b;
      --cyan: #4cc9f0;
    }

    [data-testid="stAppViewContainer"] {
      background:
        radial-gradient(circle at top left, rgba(76, 201, 240, 0.14), transparent 25%),
        radial-gradient(circle at top right, rgba(46, 213, 115, 0.14), transparent 20%),
        linear-gradient(180deg, #07111d 0%, #0b1624 100%);
      color: white;
    }

    .hero-card, .panel-card {
      background: var(--panel);
      border: 1px solid var(--panel-border);
      border-radius: 20px;
      padding: 1.25rem;
      backdrop-filter: blur(14px);
    }

    .hero-title {
      font-size: clamp(2.2rem, 4vw, 4rem);
      font-weight: 800;
      line-height: 1;
      margin-bottom: 0.5rem;
    }

    .hero-subtitle {
      color: var(--text-muted);
      font-size: 1rem;
      max-width: 42rem;
    }

    .metric-chip {
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--panel-border);
      border-radius: 16px;
      padding: 1rem;
      text-align: center;
    }

    .metric-value {
      font-size: 2rem;
      font-weight: 800;
    }

    .result-clean, .result-dirty {
      border-radius: 16px;
      padding: 1rem 1.25rem;
      margin-top: 1rem;
    }

    .result-clean {
      background: rgba(46, 213, 115, 0.12);
      border: 1px solid rgba(46, 213, 115, 0.55);
    }

    .result-dirty {
      background: rgba(255, 107, 107, 0.12);
      border: 1px solid rgba(255, 107, 107, 0.55);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="hero-card">
      <div style="letter-spacing: 0.12em; color: #4cc9f0; font-weight: 700;">AWS READY STREAMLIT APP</div>
      <div class="hero-title">EcoScan AI</div>
      <div class="hero-subtitle">
        Upload or capture an image, classify environmental waste conditions, and track detections on a live map.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

total_scans = len(st.session_state.detections)
clean_count = sum(1 for item in st.session_state.detections if "clean" in item["result"].lower())
dirty_count = total_scans - clean_count
environment_score = (clean_count / total_scans * 100) if total_scans else 0.0

metric_cols = st.columns(4)
metrics = [
    ("Total Scans", total_scans),
    ("Clean Items", clean_count),
    ("Dirty Items", dirty_count),
    ("Environment Score", f"{environment_score:.1f}%"),
]
for column, (label, value) in zip(metric_cols, metrics):
    column.markdown(
        f'<div class="metric-chip"><div>{label}</div><div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )

scan_tab, history_tab = st.tabs(["Environmental Scan", "History & Analytics"])

with scan_tab:
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.subheader("Image Input")
        source = st.radio("Choose source", ["Camera", "Upload"], horizontal=True)
        latitude = st.number_input("Latitude", value=40.7128, format="%.4f")
        longitude = st.number_input("Longitude", value=-74.0060, format="%.4f")

        uploaded_file = (
            st.camera_input("Capture image") if source == "Camera" else st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.subheader("Live Analysis")
        if uploaded_file is None:
            st.info("Waiting for image input.")
        else:
            image = Image.open(uploaded_file)
            label, confidence = predict_image(image)
            result_class = "result-clean" if "clean" in label.lower() else "result-dirty"

            st.image(image, use_container_width=True)
            st.markdown(
                f"""
                <div class="{result_class}">
                  <div style="font-size: 1.6rem; font-weight: 700;">{label}</div>
                  <div>Confidence: {confidence:.2%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button("Log Detection", use_container_width=True):
                st.session_state.detections.append(
                    {
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "result": label,
                        "confidence": f"{confidence:.2%}",
                        "lat": latitude,
                        "lon": longitude,
                    }
                )
                st.success("Detection saved to session history.")
        st.markdown("</div>", unsafe_allow_html=True)

with history_tab:
    map_col, table_col = st.columns([2, 1])

    with map_col:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.subheader("Detection Map")
        center_lat = st.session_state.detections[-1]["lat"] if st.session_state.detections else 40.7128
        center_lon = st.session_state.detections[-1]["lon"] if st.session_state.detections else -74.0060
        detection_map = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB positron")

        for detection in st.session_state.detections:
            color = "green" if "clean" in detection["result"].lower() else "red"
            folium.CircleMarker(
                location=[detection["lat"], detection["lon"]],
                radius=8,
                color=color,
                fill=True,
                fill_opacity=0.8,
                popup=f'{detection["result"]} ({detection["confidence"]})',
            ).add_to(detection_map)

        st_folium(detection_map, width=None, height=420)
        st.markdown("</div>", unsafe_allow_html=True)

    with table_col:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.subheader("History")
        if st.session_state.detections:
            history = pd.DataFrame(st.session_state.detections)
            st.dataframe(history[["time", "result", "confidence"]], use_container_width=True, hide_index=True)
        else:
            st.info("No detections logged yet.")
        st.markdown("</div>", unsafe_allow_html=True)
