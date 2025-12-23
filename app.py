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

# --- CORE ENGINE SETUP ---
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

# --- NLTK SETUP ---
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+|[^a-z\s]+', ' ', text)
    tokens = text.split()
    bias_noise = {'said', 'would', 'also', 'could', 'told', 'slated', 'municipal', 'city', 'department', 'board', 'administrative'}
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and w not in bias_noise]
    return " ".join(tokens)

# --- STYLING ---
st.set_page_config(page_title="TRUTH ENGINE", layout="wide")

st.markdown('''
<style>
    .stApp { background-color: #050505; }
    [data-testid="stSidebar"] {
        background-color: #080a0f !important;
        border-right: 2px solid #00f2ff;
    }
    .main {
        background: rgba(8, 10, 15, 0.94);
        border-top: 4px solid #00f2ff;
        padding: 40px;
        border-radius: 0 0 20px 20px;
    }
    h1, h2, h3 { color: #ffffff !important; font-family: 'Orbitron', sans-serif; }
    .stButton>button { width: 100%; background: #00f2ff; color: #080a0f; font-weight: 900; }
</style>
''', unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("## 🛰️ **TRUTH ENGINE**")
    page = st.radio("SYSTEM NAV", ["Truth Scanner", "Methodology"])
    st.divider()
    st.markdown("### 👤 **OPERATOR IDENT**")
    st.info("**Developed by Grace-96**\n\nNeural SVM Engine calibrated for high-precision detection.")
    st.caption("v2.5 Stable Build | Dec 2025")

# --- PAGE 1: TRUTH SCANNER ---
if page == "Truth Scanner":
    # Safe image loading to prevent MediaFileStorageError
    try:
        col1, col2 = st.columns([1, 4])
        with col1:
            st.image('auditor_header.jpg', width=120) # Keep your branding
        with col2:
            st.title("TRUTH ENGINE: v2.5")
            st.write("🛰️ **GLOBAL DISINFORMATION SCANNER** | STATUS: **READY**")
    except Exception:
        # Fallback if the file is missing or corrupted
        st.title("🛡️ TRUTH ENGINE: v2.5")

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

            # Result Logic
            if 0.35 < prob_true < 0.65:
                st.warning("## ⚠️ **VERDICT: AUDIT UNCERTAIN**")
                verdict = "UNCERTAIN"
            elif prob_true > 0.5:
                st.success("## **VERDICT: AUTHENTIC**")
                verdict = "AUTHENTIC"
            else:
                st.error("## **VERDICT: DECEPTIVE**")
                verdict = "DECEPTIVE"
            
            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("INTEGRITY", f"{prob_true*100:.1f}%")
            m2.metric("SKEPTICISM", f"{prob_fake*100:.1f}%")
            m3.metric("CONFIDENCE", f"{confidence*100:.1f}%")

            # Visuals
            v1, v2 = st.columns(2)
            with v1:
                st.markdown("#### ☁️ **WORD VIBRATIONS**")
                wc = WordCloud(background_color="#080a0f", colormap='cool').generate(cleaned)
                fig, ax = plt.subplots(); ax.imshow(wc); ax.axis('off'); fig.patch.set_facecolor('#080a0f')
                st.pyplot(fig)
            with v2:
                st.markdown("#### 🕵️ **FORENSIC EVIDENCE**")
                indices = vec.indices
                df = pd.DataFrame({'Term': feature_names[indices], 'Impact': vec.data * coefficients[indices]})
                st.dataframe(df.sort_values(by='Impact', ascending=False).head(8), hide_index=True)

# --- PAGE 2: METHODOLOGY ---
else:
    st.title("🔬 SYSTEM METHODOLOGY")
    st.markdown("""
    ### **Neural Audit Protocol**
    The **Truth Engine** operates on a **Calibrated SVM (Support Vector Machine)**. 
    This model evaluates the linguistic weight of terms using **TF-IDF Vectorization**.
    
    #### **Logic Thresholds:**
    * **Confident Authentic:** > 65% Integrity
    * **Confident Deceptive:** > 65% Skepticism
    * **Uncertain Zone:** 35% - 65% (Manual Review Triggered)
    
    
    
    #### **Bias Mitigation:**
    The engine filters 'noise words' like *administrative* or *municipal* to prevent false-positive deceptive verdicts on neutral reports.
    """)

st.markdown("---")
st.caption("Developed by News Integrity Auditor Labs")

