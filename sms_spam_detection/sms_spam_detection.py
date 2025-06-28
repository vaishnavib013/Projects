import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
nltk.download('punkt')
nltk.download('stopwords')

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = [ps.stem(i) for i in text if i.isalnum() and i not in stopwords.words('english') and i not in string.punctuation]
    return " ".join(y)

# Load models
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# UI
st.title("📩 SMS Spam Classifier")

input_sms = st.text_area("Enter your message:")

if st.button("Predict"):
    transformed = transform_text(input_sms)
    vector_input = vectorizer.transform([transformed])
    result = model.predict(vector_input)[0]

    if result == 1:
        st.error("⚠️ Spam Message Detected")
    else:
        st.success("✅ Not Spam")
