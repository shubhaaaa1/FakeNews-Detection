import streamlit as st
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
ps = PorterStemmer()

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

# Load saved model and vectorizer
model = pickle.load(open("C:/Users/shubh/Downloads/Trainednewsmodel.sav", 'rb'))
vectorizer = pickle.load(open("C:/Users/shubh/Downloads/vectorizer.pkl", 'rb'))

st.title("📰 Fake News Detector")

title = st.text_input("Enter News Title")
content = st.text_area("Enter News Content")

if st.button("Predict"):
    combined = title + " " + content
    cleaned = clean_text(combined)
    vect_text = vectorizer.transform([cleaned])  # ✅ Correctly transformed
    pred = model.predict(vect_text)              # ✅ Correctly used

    if pred[0] == 1:
        st.success("✅ REAL News")
    else:
        st.error("❌ FAKE News")
