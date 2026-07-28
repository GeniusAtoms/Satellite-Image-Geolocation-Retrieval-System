"""
Streamlit UI — Upload image → click Locate → see top 5 ranked results.
Run: streamlit run app.py
"""

import os
import json
import sys
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

st.set_page_config(page_title="GeoLocator", page_icon="🛰️", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { background:#080d1a !important; color:#cbd5e1; font-family:'Inter',sans-serif; }
#MainMenu, footer, header { visibility:hidden; }
section[data-testid="stSidebar"] { background:#0d1424 !important; border-right:1px solid #1e293b; }
.block-container { padding-top:2.5rem; max-width:680px; }

.title { font-family:'Space Mono',monospace; font-size:2rem; font-weight:700; color:#00ff88; margin-bottom:2px; }
.sub   { font-family:'Space Mono',monospace; font-size:0.7rem; color:#334155; text-transform:uppercase; letter-spacing:0.2em; margin-bottom:2rem; }

.rank-card {
    background:#0a1020; border:1px solid #1a3a5c;
    border-radius:10px; padding:1rem 1.4rem; margin-bottom:0.55rem;
    display:flex; align-items:center; gap:1.4rem;
}
.rank-card.top { border-color:#00ff88; background:#020f08; }

.rank-num { font-family:'Space Mono',monospace; font-size:1.6rem; font-weight:700; color:#1e3a5f; min-width:2rem; text-align:center; }
.rank-card.top .rank-num { color:#00ff88; }

.coords { flex:1; }
.coord-row { display:flex; gap:2rem; }
.coord-item {}
.clabel { font-family:'Space Mono',monospace; font-size:0.6rem; color:#334155; text-transform:uppercase; letter-spacing:0.18em; }
.cval   { font-family:'Space Mono',monospace; font-size:1.25rem; font-weight:700; color:#e2e8f0; }
.rank-card.top .cval { color:#00ff88; }

.stButton > button[kind="primary"] {
    background:#00ff88 !important; color:#080d1a !important;
    border:none !important; font-family:'Space Mono',monospace !important;
    font-weight:700 !important; font-size:1rem !important;
    border-radius:8px !important; height:3rem;
}
.stButton > button[kind="primary"]:hover { background:#00e87a !important; }

.chip-ok   { background:#021a0e; color:#00ff88; border:1px solid #00c96a; padding:3px 12px; border-radius:4px; font-size:0.72rem; font-family:'Space Mono',monospace; }
.chip-warn { background:#1a1200; color:#f59e0b; border:1px solid #d97706; padding:3px 12px; border-radius:4px; font-size:0.72rem; font-family:'Space Mono',monospace; }
</style>
""", unsafe_allow_html=True)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def index_ready(d):
    return os.path.exists(os.path.join(d,"geo_index.faiss")) and os.path.exists(os.path.join(d,"metadata.json"))

def meta_count(d):
    with open(os.path.join(d,"metadata.json")) as f: return len(json.load(f))

# ── Sidebar (index management) ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🗂 Index")
    index_dir = st.text_input("Index directory", value="./index_store_all")
    ready = index_ready(index_dir)
    if ready:
        st.markdown(f'<span class="chip-ok">✓ {meta_count(index_dir):,} images indexed</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="chip-warn">⚠ No index found</span>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### ⚙ Build Index")
    dataset_path = st.text_input("Dataset folder", placeholder="/path/to/images")
    index_type   = st.selectbox("Index type", ["flat","ivf","hnsw"])
    batch_size   = st.slider("Batch size", 8, 128, 32, step=8)
    if st.button("Build Index", use_container_width=True, type="primary"):
        if not dataset_path:
            st.error("Enter dataset folder.")
        elif not os.path.isdir(dataset_path):
            st.error(f"Not found: {dataset_path}")
        else:
            with st.spinner("Building index..."):
                try:
                    sys.path.insert(0, THIS_DIR)
                    from build_index import build
                    build(dataset_path, index_dir, batch_size=batch_size, index_type=index_type)
                    st.success("Done.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown('<div class="title">🛰 GeoLocator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Upload a satellite image · get top 5 locations</div>', unsafe_allow_html=True)

uploaded = st.file_uploader("", type=["png","jpg","jpeg"], label_visibility="collapsed")

if uploaded:
    if st.session_state.get("last_file") != uploaded.name:
        st.session_state.pop("results", None)
        st.session_state["last_file"] = uploaded.name

locate_clicked = st.button("Locate", use_container_width=True, type="primary", disabled=not uploaded or not ready)

if locate_clicked and uploaded and ready:
    with st.spinner("Searching..."):
        try:
            suffix = Path(uploaded.name).suffix or ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name
            sys.path.insert(0, THIS_DIR)
            from locate import locate
            st.session_state["results"] = locate(tmp_path, index_dir)
            os.unlink(tmp_path)
        except Exception as e:
            st.error(str(e))

if "results" in st.session_state:
    st.markdown("---")
    for r in st.session_state["results"]:
        rank    = r["rank"]
        lat     = r["latitude"]
        lon     = r["longitude"]
        lat_dir = "N" if lat >= 0 else "S"
        lon_dir = "E" if lon >= 0 else "W"
        cls     = "rank-card top" if rank == 1 else "rank-card"
        label   = "★" if rank == 1 else str(rank)

        st.markdown(f"""
        <div class="{cls}">
            <div class="rank-num">{label}</div>
            <div class="coords">
                <div class="coord-row">
                    <div class="coord-item">
                        <div class="clabel">Latitude</div>
                        <div class="cval">{lat:+.6f} {lat_dir}</div>
                    </div>
                    <div class="coord-item">
                        <div class="clabel">Longitude</div>
                        <div class="cval">{lon:+.6f} {lon_dir}</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)