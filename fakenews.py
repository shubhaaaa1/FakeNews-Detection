import streamlit as st
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download stopwords
nltk.download('stopwords')
ps = PorterStemmer()

# --- Page Config ---
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# --- Custom CSS Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title and Intro ---
st.markdown("<h1 style='text-align: center; color: navy;'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: gray;'>Powered by Logistic Regression and NLP</h5>", unsafe_allow_html=True)
st.markdown("---")

# --- Load Model and Vectorizer ---
model = pickle.load(open("model.pkl", 'rb'))
vectorizer = pickle.load(open("vectorizer.pkl", 'rb'))

# --- Text Cleaning Function ---
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

# --- User Input ---
with st.container():
   title = st.text_input("📝 Enter News Title", placeholder="e.g. India wins T20 World Cup 2024")
   content = st.text_area("📰 Enter News Content", placeholder="e.g. In a stunning finale, India lifted...")


# --- Predict Button ---
if st.button("🔍 Predict"):
    if title.strip() == "" or content.strip() == "":
        st.warning("⚠️ Please enter both title and content.")
    else:
        combined = title + " " + content
        cleaned = clean_text(combined)
        vect_text = vectorizer.transform([cleaned])
        pred = model.predict(vect_text)
        proba = model.predict_proba(vect_text)[0]
        confidence = max(proba) * 100
        st.info(f"🧠 Confidence: {confidence:.2f}%")


        if pred[0] == 1:
            st.success("✅ This looks like **REAL** news!")
        else:
            st.error("🚫 This appears to be **FAKE** news.")
     
# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Made with ❤️ by Shubham</p>", unsafe_allow_html=True)
