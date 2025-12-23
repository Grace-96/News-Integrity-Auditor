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

# --- SIDEBAR NAV & IDENTITY ---
with st.sidebar:
    # We use Markdown instead of st.image to prevent storage errors
    st.markdown("## 🛰️ **CORE SYSTEM**")
    st.markdown("---")
    
    st.markdown("### 🔭 **SYSTEM NAV**")
    # One single radio button to control your pages
    page = st.radio("Select Interface:", ["Truth Scanner", "Methodology"], label_visibility="collapsed")
    
    st.divider()
    
    st.markdown("### 👤 **OPERATOR IDENT**")
    st.info("""
    **Developed by Grace-96**
    
    Neural SVM Engine calibrated for high-precision disinformation detection.
    """)
    
    st.caption("v2.5 Stable Build | Dec 2025")
    st.caption("v2.5 Stable Build | Dec 2025")

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
    <style>
    /* 1. Target the Sidebar specifically */
    [data-testid="stSidebar"] {
        background-color: #080a0f !important; /* Deep Void Black */
        border-right: 2px solid #00f2ff; /* Neon Cyan Glow */
    }
    
    /* 2. Style the text inside the Sidebar */
    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-family: 'Orbitron', sans-serif;
    }

    /* 3. Make the Info box look like a glowing terminal */
    [data-testid="stSidebar"] .stAlert {
        background-color: #0d1117;
        color: #00f2ff;
        border: 1px solid #00f2ff;
    }
    
    </style>
    ''', unsafe_allow_html=True)

# --- CONFIGURATION CONSTANTS ---
# Wider thresholds to catch biased predictions
UNCERTAINTY_THRESHOLD_LOW = 0.35 
UNCERTAINTY_THRESHOLD_HIGH = 0.65 

# --- CORE ENGINE ---
REPO_ID = "Grace-96/News-Integrity-Auditor-Models"

# Initialize NLTK components 
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
except LookupError:
    st.error("FATAL ERROR: NLTK language data is missing.")
    st.stop()

def clean_text(text):
    """Deep cleaning to remove biased common words."""
    text = text.lower()
    # Remove URLs and non-alphanumeric junk
    text = re.sub(r'http\S+|www\S+|https\S+|[^a-z\s]+', ' ', text)
    
    tokens = text.split()
    # Words that often cause accidental 'Fake' bias in simple models
    bias_noise = {'said', 'would', 'also', 'could', 'told', 'slated', 'municipal', 'city', 'department'}
    
    tokens = [w for w in tokens if w not in stop_words and w not in bias_noise]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    
    return " ".join(tokens)

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

header_col1, header_col2 = st.columns([1, 4])
with header_col1:
    try:
        st.image(Image.open('r.jpg'), width=150)
    except: st.markdown("# 🛡️")
with header_col2:
    st.title("TRUTH ENGINE: v2.5")
    st.write("🛰️ **GLOBAL DISINFORMATION SCANNER** | STATUS: **READY**")

st.markdown("---")

st.markdown("### 🧬 **LINGUISTIC FEED: INPUT TARGET DATA**")
raw_input = st.text_area("", placeholder="Paste article content here...", height=250)

if st.button("INITIATE TRUTH SCAN"):
    if raw_input.strip():
        with st.status("📡 **SCANNING NEURAL MARKERS...**", expanded=True) as status:
            cleaned = clean_text(raw_input)
            vec = tfidf.transform([cleaned])
            probs = model.predict_proba(vec)[0]
            
            prob_fake, prob_true = probs[0], probs[1]
            confidence = max(probs)
            
            time.sleep(1.5)
            status.update(label="SCAN COMPLETE", state="complete")

        # Result Logic with Uncertainty Warning
        if UNCERTAINTY_THRESHOLD_LOW < prob_true < UNCERTAINTY_THRESHOLD_HIGH:
            st.warning("## ⚠️ **VERDICT: AUDIT UNCERTAIN**")
            st.info("The system could not confidently classify this article. Manual forensic review required.")
            verdict = "UNCERTAIN"
        elif np.argmax(probs) == 1:
            st.success("## **VERDICT: AUTHENTIC**")
            verdict = "AUTHENTIC"
        else:
            st.error("## **VERDICT: DECEPTIVE**")
            verdict = "DECEPTIVE"
        
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("INTEGRITY SCORE", f"{prob_true*100:.1f}%")
        with m_col2:
            st.metric("SKEPTICISM LEVEL", f"{prob_fake*100:.1f}%")
        with m_col3:
            st.metric("ENGINE CONFIDENCE", f"{confidence*100:.1f}%")

        st.markdown("---")
        
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            st.markdown("#### ☁️ **WORD VIBRATIONS**")
            wc = WordCloud(background_color="#080a0f", colormap='cool', width=600, height=300).generate(cleaned)
            fig, ax = plt.subplots(); ax.imshow(wc, interpolation="bilinear"); ax.axis('off')
            fig.patch.set_facecolor('#080a0f')
            st.pyplot(fig)
            
        with v_col2:
            st.markdown("#### 🕵️ **FORENSIC EVIDENCE**")
            present_indices = vec.indices
            df_contrib = pd.DataFrame({
                'Term': feature_names[present_indices],
                'Impact': vec.data * coefficients[present_indices]
            }).sort_values(by='Impact', key=abs, ascending=False).head(8)
            
            st.dataframe(df_contrib, hide_index=True, use_container_width=True,
                         column_config={"Impact": st.column_config.NumberColumn(format="%.4f")})

            st.markdown("---")
            report_data = (
                f"TRUTH ENGINE INTEL REPORT\n"
                f"==========================\n"
                f"TIMESTAMP: {time.ctime()}\n"
                f"VERDICT: {verdict}\n"
                f"ENGINE CONFIDENCE: {confidence*100:.2f}%\n"
                f"INTEGRITY SCORE: {prob_true*100:.1f}%\n"
                f"SKEPTICISM LEVEL: {prob_fake*100:.1f}%\n"
            )
            st.download_button(
                label="📩 DOWNLOAD INTELLIGENCE REPORT",
                data=report_data,
                file_name=f"Audit_Report_{int(time.time())}.txt",
                mime="text/plain"
            )

    else:
        st.toast("⚠️ Input required for scanning.", icon="🚨")

st.markdown("---")
st.caption("Developed by News Integrity Auditor Labs | Proprietary Neural Engine")







