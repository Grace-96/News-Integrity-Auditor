import streamlit as st
import joblib
import numpy as np
import os
import sys
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
import time
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt 
import base64 

# ------------------------------------------------------------------
# UTILITY FUNCTIONS: BASE64 ENCODING AND TEXT CLEANING
# ------------------------------------------------------------------
def get_base64_of_bin_file(bin_file):
    """Encodes local file to Base64 string for CSS injection."""
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None 

def set_bg_image(file_path):
    """Injects CSS to set the Base64 encoded image as the page background."""
    bin_str = get_base64_of_bin_file(file_path)
    mime_type = "image/jpeg" if file_path.lower().endswith(('.jpg', '.jpeg')) else "image/png"
    
    if bin_str:
        page_bg_img = f''' 
        <style>
        .stApp {{
            background-image: url("data:{mime_type};base64,{bin_str}");
            background-size: cover;
            background-attachment: fixed;
        }}
        .main > div {{ 
            background-color: rgba(14, 17, 23, 0.85);
            padding: 20px;
            border-radius: 5px;
        }}
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)
    # Note: If image is not found, we rely on the primary CSS block for a solid dark background.

# Initialize NLTK components 
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
except LookupError:
    st.error("FATAL ERROR: NLTK language data is missing. Cannot proceed with text processing.")
    st.stop()


def clean_text(text):
    """Preprocessing function integrated directly into the app for robustness."""
    
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+|[^a-z0-9\s]+', ' ', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    
    return " ".join(tokens)


# --- CONFIGURATION CONSTANTS ---
UNCERTAINTY_THRESHOLD_LOW = 0.45 
UNCERTAINTY_THRESHOLD_HIGH = 0.55 
# --- END CONFIGURATION ---

# ----------------------------------------------------
# 2. MODEL AND LOGGING FUNCTIONS (CLOUD COMPATIBLE)
# ----------------------------------------------------
from huggingface_hub import hf_hub_download
import joblib

# Ensure this matches your Hugging Face username and repo name exactly
REPO_ID = "Grace-96/News-Integrity-Auditor-Models" 

@st.cache_resource 
def load_all_assets():
    try:
        # This downloads the files from Hugging Face instead of looking for a local 'models/' folder
        tfidf_path = hf_hub_download(repo_id=REPO_ID, filename="tfidf_vectorizer.pkl")
        model_path = hf_hub_download(repo_id=REPO_ID, filename="calibrated_svm.pkl")
        feature_path = hf_hub_download(repo_id=REPO_ID, filename="svm_features.pkl")
        
        return joblib.load(tfidf_path), joblib.load(model_path), joblib.load(feature_path)
    except Exception as e:
        st.error(f"FATAL ERROR: Could not fetch models from Hugging Face. Error: {e}")
        st.stop()

# Load the variables
tfidf, model, feature_data = load_all_assets()

# Set up the features for your visuals
feature_names = np.array(feature_data['feature_names'])
coefficients = np.array(feature_data['coefficients'])


# ----------------------------------------------------
# 3. STREAMLIT UI AND LOGIC (FULL FUNCTIONALITY)
# ----------------------------------------------------

st.set_page_config(page_title="Integrity Auditor", layout="centered")

# --- CUSTOM CSS INJECTION (SOLID DARK BACKGROUND) ---
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0E1117; 
        color: #FAFAFA; 
    }
    .stButton>button {
        background-color: #DC143C; 
        color: white;
    }
    p, h1, h2, h3, h4, .stMarkdown {
        color: #FAFAFA !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- EXECUTE BACKGROUND IMAGE (Will fallback to solid dark if file not found) ---
set_bg_image('bg_image.jpg') 


# --- IMPACTFUL HEADINGS AND CATCHPHRASE ---
st.title("🛡️ **AUTHENTICITY VERIFICATION SYSTEM**") # New, bolder title
st.subheader("Detecting Disinformation. Delivering Fact.") # New Catchphrase

try:
    # Use the corrected parameter: use_container_width
    image = Image.open('app/auditor_header.jpg') 
    st.image(image, use_container_width=True) 
except FileNotFoundError:
    st.subheader("Input Protocol Active")
    
st.markdown("---")

with st.expander("▶️ **SYSTEM OVERVIEW AND PROTOCOL**"): # New Heading
    st.markdown("""
        This algorithm instantly verifies the **integrity** of news articles by analyzing core linguistic patterns.
        It uses a highly accurate, calibrated SVM model to detect subtle language cues typical of fabricated or unreliable sources.
        
        * **Methodology:** Calibrated Support Vector Machine (SVM) on TF-IDF features.
        * **Compliance:** Developed for high-performance portfolio demonstration.
    """)
    
st.markdown("---")

st.markdown("### 📝 **ARTICLE SUBMISSION PORTAL**") # New Heading
text = st.text_area("TARGET CONTENT: Paste article text for truth audit:", height=250)

if st.button("VERIFY TRUTH", type="primary"):
    if not text.strip():
        st.warning("Input required. Please paste an article.")
        st.stop()

    with st.spinner("Processing text for verification..."):
        # 1. Prediction Setup
        cleaned = clean_text(text)
        vec = tfidf.transform([cleaned])
        probabilities = model.predict_proba(vec)[0]
        prob_fake = probabilities[0]
        prob_true = probabilities[1]
        
        predicted_class = np.argmax(probabilities)
        predicted_name = "TRUE" if predicted_class == 1 else "FALSE" 
        confidence_score = max(prob_fake, prob_true)
        
        
        # --- 2. AUDIT RESULT DISPLAY ---
        st.markdown("---")
        
        if UNCERTAINTY_THRESHOLD_LOW < prob_true < UNCERTAINTY_THRESHOLD_HIGH:
            log_prediction(text, predicted_name, confidence_score, 'UNCERTAIN')
            
            with st.container(border=True):
                st.warning("## ⚠️ **AUDIT UNCERTAIN: MANUAL REVIEW REQUIRED**")
                st.markdown(f"**Verdict:** The system could not confidently classify the article.")
                st.caption(f"Certainty Level: {confidence_score*100:.2f}%")
            
        elif predicted_class == 1:
            log_prediction(text, predicted_name, confidence_score, 'HIGH_CONFIDENCE')
            
            with st.container(border=True):
                st.success("## **✅ AUDIT RESULT: TRUE**")
                st.subheader(f"Verification Score: **:green[{prob_true*100:.2f}%]**")
            
        else: 
            log_prediction(text, predicted_name, confidence_score, 'HIGH_CONFIDENCE')

            with st.container(border=True):
                st.error("## **🛑 AUDIT RESULT: FALSE**")
                st.subheader(f"Verification Score: **:red[{prob_fake*100:.2f}%]**")

        # --- 3. EVIDENTIARY ANALYSIS ---
        st.markdown("### 📊 **EVIDENTIARY ANALYSIS AND VISUALS**") # New Heading
        st.markdown("---")
        st.info("The summary below provides a deep dive into the language structure that drove the verdict:")
        
        
        # --- WORD CLOUD AND DISTRIBUTION ---
        col_chart1, col_chart2 = st.columns(2) 
        
        # --- Word Cloud ---
        with col_chart1:
            st.markdown("#### ☁️ **Linguistic Pattern Cloud**") # New Heading
            wordcloud = WordCloud(
                width=400, 
                height=200, 
                background_color="#1C2833", 
                colormap='viridis',
                min_font_size=8,
                max_words=100
            ).generate(cleaned)
            
            fig_wc, ax_wc = plt.subplots(figsize=(4, 2))
            ax_wc.imshow(wordcloud, interpolation="bilinear")
            ax_wc.axis("off") 
            st.pyplot(fig_wc)

        # --- Word Distribution Chart ---
        with col_chart2:
            st.markdown("#### 📈 **Top Token Frequency**") # New Heading
            
            words = cleaned.split()
            word_counts = pd.Series(words).value_counts().head(10)
            
            fig_bar, ax_bar = plt.subplots(figsize=(4, 2))
            
            ax_bar.bar(word_counts.index, word_counts.values, color='#DC143C') 
            ax_bar.set_title('')
            ax_bar.tick_params(axis='x', rotation=45, labelsize=7, colors='#FAFAFA')
            ax_bar.tick_params(axis='y', labelsize=7, colors='#FAFAFA')
            
            ax_bar.set_facecolor('#1C2833')
            fig_bar.patch.set_facecolor('#1C2833')
            for spine in ax_bar.spines.values():
                spine.set_edgecolor('#262730')

            plt.tight_layout()
            st.pyplot(fig_bar)

        # --- Influence Table (The CORE Proof) ---
        st.markdown("#### 🔑 **Key Feature Influence Report**") # New Heading
        top_features = get_top_features_for_text(vec, feature_names, coefficients, top_n=10)
        
        if top_features:
            col_word, col_score, col_push = st.columns([1, 1, 3])
            col_word.markdown("**Term**")
            col_score.markdown("**Weight**")
            col_push.markdown("**System Bias**") # New Label

            for item in top_features:
                score = float(item['score'])
                
                color = "green" if score > 0 else "red"
                label = "TOWARDS TRUE" if score > 0 else "TOWARDS FALSE"
                
                col_word, col_score, col_push = st.columns([1, 1, 3])
                with col_word:
                    st.markdown(f"**{item['word']}**")
                with col_score:
                    st.markdown(f"<span style='color:{color}'>{score:.4f}</span>", unsafe_allow_html=True)
                with col_push:
                    st.markdown(f"**{label}**")

        st.markdown("---")
        st.caption("Protocol: High-Confidence Machine Learning Audit. Code for demonstration purposes.")
