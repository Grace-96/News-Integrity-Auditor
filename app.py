import streamlit as st
import joblib
import numpy as np
import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import time
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt 
import base64 
from huggingface_hub import hf_hub_download

# --- CYBER-INTELLIGENCE STYLING ---
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except: return None 

def apply_intelligence_theme(file_path):
    bin_str = get_base64_of_bin_file(file_path)
    bg_css = f'background-image: url("data:image/jpg;base64,{bin_str}");' if bin_str else "background-color: #050505;"
    st.markdown(f'''
    <style>
    .stApp {{
        {bg_css}
        background-size: cover;
        background-attachment: fixed;
    }}
    .main {{
        background: rgba(8, 10, 15, 0.94); /* Deep Void Black */
        border-top: 4px solid #00f2ff;
        padding: 40px;
        border-radius: 0 0 20px 20px;
    }}
    h1, h2 {{
        color: #ffffff !important;
        font-family: 'Orbitron', sans-serif;
        letter-spacing: 3px;
        text-transform: uppercase;
    }}
    .stTextArea textarea {{
        background-color: #0d1117 !important;
        color: #00f2ff !important;
        border: 1px solid #30363d !important;
    }}
    .stButton>button {{
        width: 100%;
        background: #00f2ff;
        color: #080a0f;
        border-radius: 5px;
        font-weight: 900;
        text-transform: uppercase;
        border: none;
        transition: 0.5s;
    }}
    .stButton>button:hover {{
        background: #ff0055;
        color: white;
        box-shadow: 0 0 25px #ff0055;
    }}
    </style>
    ''', unsafe_allow_html=True)

# --- CORE ENGINE ---
REPO_ID = "Grace-96/News-Integrity-Auditor-Models"
@st.cache_resource
def boot_system():
    try:
        t_p = hf_hub_download(repo_id=REPO_ID, filename="tfidf_vectorizer.pkl")
        m_p = hf_hub_download(repo_id=REPO_ID, filename="calibrated_svm.pkl")
        f_p = hf_hub_download(repo_id=REPO_ID, filename="svm_features.pkl")
        return joblib.load(t_p), joblib.load(m_p), joblib.load(f_p)
    except Exception as e:
        st.error(f"SYSTEM FAILURE: {e}"); st.stop()

tfidf, model, f_data = boot_system()
feature_names = np.array(f_data['feature_names'])
coefficients = np.array(f_data['coefficients'])

# --- UI CONTENT ---
st.set_page_config(page_title="TRUTH ENGINE", layout="wide")
apply_intelligence_theme('background.jpg')

# Top Navigation Style Header
header_col1, header_col2 = st.columns([1, 4])
with header_col1:
    try:
        st.image(Image.open('auditor_header.jpg'), width=150)
    except: st.markdown("# 🛡️")
with header_col2:
    st.title("TRUTH ENGINE: v2.5")
    st.write("🛰️ **GLOBAL DISINFORMATION SCANNER** | STATUS: **READY**")

st.markdown("---")

# Main Input Section
st.markdown("### 🧬 **LINGUISTIC FEED: INPUT TARGET DATA**")
raw_input = st.text_area("", placeholder="Paste article content or suspected propaganda here...", height=250)

if st.button("INITIATE TRUTH SCAN"):
    if raw_input.strip():
        with st.status("📡 **SCANNING NEURAL MARKERS...**", expanded=True) as status:
            # Processing
            cleaned = re.sub(r'[^a-z0-9\s]', ' ', raw_input.lower())
            vec = tfidf.transform([cleaned])
            probs = model.predict_proba(vec)[0]
            
            # Catchy Verdicts
            is_true = np.argmax(probs) == 1
            verdict = "AUTHENTIC" if is_true else "DECEPTIVE"
            confidence = max(probs)
            
            time.sleep(1.5)
            status.update(label="SCAN COMPLETE", state="complete")

        # Visual Dashboard
        st.markdown(f"## **VERDICT: {verdict}**")
        
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("INTEGRITY SCORE", f"{probs[1]*100:.1f}%")
        with m_col2:
            st.metric("SKEPTICISM LEVEL", f"{probs[0]*100:.1f}%")
        with m_col3:
            st.metric("ENGINE CONFIDENCE", f"{confidence*100:.1f}%")

        st.markdown("---")
        
        # Fresh Visuals
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            st.markdown("#### ☁️ **WORD VIBRATIONS**")
            # Word cloud with custom dark colors
            wc = WordCloud(background_color="#080a0f", colormap='cyan', width=600, height=300).generate(cleaned)
            fig, ax = plt.subplots(); ax.imshow(wc, interpolation="bilinear"); ax.axis('off')
            fig.patch.set_facecolor('#080a0f')
            st.pyplot(fig)
            
        with v_col2:
            st.markdown("#### 🕵️ **FORENSIC EVIDENCE**")
            # Get real top features
            present_indices = vec.indices
            df_contrib = pd.DataFrame({
                'Term': feature_names[present_indices],
                'Impact': vec.data * coefficients[present_indices]
            }).sort_values(by='Impact', key=abs, ascending=False).head(8)
            
            st.dataframe(df_contrib, hide_index=True, use_container_width=True)

    else:
        st.toast("⚠️ Input required for scanning.", icon="🚨")

st.markdown("---")
st.caption("Developed by News Integrity Auditor Labs | Proprietary Neural Engine")
