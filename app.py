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
import time
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt 
import base64 
from huggingface_hub import hf_hub_download

# ------------------------------------------------------------------
# 1. UTILITY FUNCTIONS: BASE64 ENCODING AND TEXT CLEANING
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
    """Injects CSS to set the background with a 'Glass' layer to protect text readability."""
    bin_str = get_base64_of_bin_file(file_path)
    if bin_str:
        st.markdown(f''' 
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{bin_str}");
            background-size: cover;
            background-attachment: fixed;
        }}
        /* Creates a semi-transparent dark layer over the background */
        .main {{
            background: rgba(14, 17, 23, 0.75); 
            border-radius: 15px;
            padding: 20px;
        }}
        </style>
        ''', unsafe_allow_html=True)

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
    """Linguistic preprocessing for the model."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+|[^a-z0-9\s]+', ' ', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

# --- CONFIGURATION ---
UNCERTAINTY_THRESHOLD_LOW = 0.45 
UNCERTAINTY_THRESHOLD_HIGH = 0.55 
REPO_ID = "Grace-96/News-Integrity-Auditor-Models" 

# ----------------------------------------------------
# 2. MODEL LOADING AND ANALYSIS FUNCTIONS
# ----------------------------------------------------
@st.cache_resource
def load_all_assets():
    try:
        # Fetches models from Hugging Face instead of local storage
        tfidf_path = hf_hub_download(repo_id=REPO_ID, filename="tfidf_vectorizer.pkl")
        model_path = hf_hub_download(repo_id=REPO_ID, filename="calibrated_svm.pkl")
        feature_path = hf_hub_download(repo_id=REPO_ID, filename="svm_features.pkl")
        return joblib.load(tfidf_path), joblib.load(model_path), joblib.load(feature_path)
    except Exception as e:
        st.error(f"FATAL ERROR: Could not fetch models from Hugging Face. Error: {e}")
        st.stop()

# Initialize variables
tfidf, model, feature_data = load_all_assets()
feature_names = np.array(feature_data['feature_names'])
coefficients = np.array(feature_data['coefficients'])

def log_prediction(text, prediction_name, confidence_score, status):
    """Logs verification metadata to a CSV."""
    LOG_FILE = "live_feedback_log.csv"
    file_exists = os.path.isfile(LOG_FILE)
    log_entry = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'input_text': text[:500] + "...",
        'prediction': prediction_name,
        'confidence': f"{confidence_score:.4f}",
        'status': status
    }
    pd.DataFrame([log_entry]).to_csv(LOG_FILE, mode='a', header=not file_exists, index=False)

def get_top_features_for_text(text_vector, feature_names, coefficients, top_n=10):
    """Calculates term influence weights."""
    present_indices = text_vector.indices
    df_contrib = pd.DataFrame({
        'word': feature_names[present_indices],
        'score': text_vector.data * coefficients[present_indices]
    })
    return df_contrib.reindex(df_contrib['score'].abs().sort_values(ascending=False).index).head(top_n).to_dict('records')

# ----------------------------------------------------
# 3. STREAMLIT UI LAYOUT
# ----------------------------------------------------
st.set_page_config(page_title="Integrity Auditor", layout="centered")

# Set Background Image (ensure this file exists in your main GitHub folder)
set_bg_image('background.jpg') 

st.title("🛡️ **AUTHENTICITY VERIFICATION SYSTEM**")
st.subheader("Detecting Disinformation. Delivering Fact.")

# Display Header Image
try:
    # Looks for 'auditor_header.jpg' in the main folder
    image = Image.open('auditor_header.jpg') 
    st.image(image, use_container_width=True) 
except FileNotFoundError:
    st.info("Input Protocol Active")
    
st.markdown("---")

with st.expander("▶️ **SYSTEM OVERVIEW AND PROTOCOL**"):
    st.markdown("""
        This algorithm instantly verifies news articles by analyzing core linguistic patterns.
        * **Methodology:** Calibrated Support Vector Machine (SVM) on TF-IDF features.
    """)
    
st.markdown("### 📝 **ARTICLE SUBMISSION PORTAL**")
text = st.text_area("TARGET CONTENT: Paste article text for truth audit:", height=250)

if st.button("VERIFY TRUTH", type="primary"):
    if not text.strip():
        st.warning("Input required. Please paste an article.")
        st.stop()

    with st.spinner("Processing text for verification..."):
        cleaned = clean_text(text)
        vec = tfidf.transform([cleaned])
        probabilities = model.predict_proba(vec)[0]
        prob_fake, prob_true = probabilities[0], probabilities[1]
        
        predicted_class = np.argmax(probabilities)
        predicted_name = "TRUE" if predicted_class == 1 else "FALSE" 
        confidence_score = max(prob_fake, prob_true)
        
        st.markdown("---")
        
        if UNCERTAINTY_THRESHOLD_LOW < prob_true < UNCERTAINTY_THRESHOLD_HIGH:
            log_prediction(text, predicted_name, confidence_score, 'UNCERTAIN')
            st.warning(f"## ⚠️ **AUDIT UNCERTAIN** \n Certainty Level: {confidence_score*100:.2f}%")
        elif predicted_class == 1:
            log_prediction(text, predicted_name, confidence_score, 'HIGH_CONFIDENCE')
            st.success(f"## **✅ AUDIT RESULT: TRUE** \n Verification Score: **{prob_true*100:.2f}%**")
        else: 
            log_prediction(text, predicted_name, confidence_score, 'HIGH_CONFIDENCE')
            st.error(f"## **🛑 AUDIT RESULT: FALSE** \n Verification Score: **{prob_fake*100:.2f}%**")

        st.markdown("### 📊 **EVIDENTIARY ANALYSIS**")
        col_chart1, col_chart2 = st.columns(2) 
        
        with col_chart1:
            wordcloud = WordCloud(width=400, height=200, background_color="#1C2833", colormap='viridis').generate(cleaned)
            fig_wc, ax_wc = plt.subplots()
            ax_wc.imshow(wordcloud, interpolation="bilinear")
            ax_wc.axis("off") 
            st.pyplot(fig_wc)

        with col_chart2:
            words = cleaned.split()
            word_counts = pd.Series(words).value_counts().head(10)
            fig_bar, ax_bar = plt.subplots()
            ax_bar.bar(word_counts.index, word_counts.values, color='#DC143C') 
            ax_bar.tick_params(axis='x', rotation=45, labelsize=7, colors='white')
            ax_bar.set_facecolor('#1C2833')
            fig_bar.patch.set_facecolor('#1C2833')
            st.pyplot(fig_bar)

        st.markdown("#### 🔑 **Key Feature Influence Report**")
        top_features = get_top_features_for_text(vec, feature_names, coefficients, top_n=10)
        
        if top_features:
            for item in top_features:
                score = float(item['score'])
                color = "green" if score > 0 else "red"
                label = "TOWARDS TRUE" if score > 0 else "TOWARDS FALSE"
                st.markdown(f"**{item['word']}**: <span style='color:{color}'>{score:.4f}</span> — {label}", unsafe_allow_html=True)

st.caption("Protocol: High-Confidence Machine Learning Audit.")
