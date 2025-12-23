import streamlit as st
import joblib
import numpy as np
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
import shap #
from streamlit_shap import st_shap #

# --- CYBER-INTELLIGENCE STYLING ---
def apply_intelligence_theme(file_path):
    st.markdown(f'''
    <style>
    .stApp {{ background-color: #050505; }}
    .main {{
        background: rgba(8, 10, 15, 0.94);
        border-top: 4px solid #00f2ff;
        padding: 40px;
        border-radius: 0 0 20px 20px;
    }}
    h1, h2, h3, h4 {{ color: #ffffff !important; font-family: 'Orbitron', sans-serif; }}
    .stButton>button {{
        width: 100%; background: #00f2ff; color: #080a0f; font-weight: 900;
    }}
    </style>
    ''', unsafe_allow_html=True)

# --- CONFIGURATION ---
UNCERTAINTY_THRESHOLD_LOW = 0.35 
UNCERTAINTY_THRESHOLD_HIGH = 0.65 

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

# --- UI CONTENT ---
st.set_page_config(page_title="TRUTH ENGINE", layout="wide")
apply_intelligence_theme(None)

st.title("TRUTH ENGINE: v2.5")
raw_input = st.text_area("📡 LINGUISTIC FEED", placeholder="Paste article content...", height=200)

if st.button("INITIATE TRUTH SCAN"):
    if raw_input.strip():
        with st.status("📡 SCANNING NEURAL MARKERS...", expanded=True):
            # Preprocessing
            cleaned = raw_input.lower()
            vec = tfidf.transform([cleaned])
            probs = model.predict_proba(vec)[0]
            prob_true = probs[1]
            
            # Initialize SHAP for Forensic Plot
            explainer = shap.LinearExplainer(model.calibrated_classifiers_[0].base_estimator, vec)
            shap_values = explainer.shap_values(vec)
            
            time.sleep(1)

        # Verdict Logic
        if UNCERTAINTY_THRESHOLD_LOW < prob_true < UNCERTAINTY_THRESHOLD_HIGH:
            st.warning("## ⚠️ VERDICT: AUDIT UNCERTAIN")
        elif prob_true > 0.5:
            st.success(f"## **VERDICT: AUTHENTIC ({prob_true*100:.1f}%)**")
        else:
            st.error(f"## **VERDICT: DECEPTIVE ({probs[0]*100:.1f}%)**")

        st.markdown("---")
        
        # --- NEW SHAP FORENSIC CHART ---
        st.markdown("#### 🕵️ **NEURAL LINGUISTIC FORENSICS**")
        st.write("Visualizing how specific terms influenced the verdict (Red = Authentic, Blue = Deceptive).")
        # Displaying the SHAP Force Plot
        st_shap(shap.force_plot(explainer.expected_value, shap_values, vec.toarray(), feature_names=feature_names))

        # Original Visuals
        vcol1, vcol2 = st.columns(2)
        with vcol1:
            st.markdown("#### ☁️ WORD VIBRATIONS")
            wc = WordCloud(background_color="#080a0f", colormap='cool').generate(cleaned)
            fig, ax = plt.subplots(); ax.imshow(wc); ax.axis('off')
            fig.patch.set_facecolor('#080a0f')
            st.pyplot(fig)
        with vcol2:
            st.markdown("#### 🧪 DATA INFLUENCE")
            df_contrib = pd.DataFrame({
                'Term': feature_names[vec.indices],
                'Impact': vec.data * model.calibrated_classifiers_[0].base_estimator.coef_[0][vec.indices]
            }).sort_values(by='Impact', ascending=False).head(10)
            st.bar_chart(df_contrib, x='Term', y='Impact', color="#00f2ff")

st.caption("Developed by News Integrity Auditor Labs")
