# Imports================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import nltk
import re
from nltk.stem import PorterStemmer

# 
# Téléchargement des ressources nécessaires (une seule fois)
nltk.download('punkt')
nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words("english"))

# Charger les models=========================

lg = pickle.load(open("logistic_model.pkl", "rb"))
lb = pickle.load(open("Label_Encoder.pkl", "rb"))
tfidf_vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Création du fonction de nétoyage du text :

def clean_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)

# Création du fonction de prédiction d'emotion:

def predict_emotion(input_text):
    cleaned_text = clean_text(input_text)
    input_vectorized = tfidf_vectorizer.transform([cleaned_text])

    predicted_label = lg.predict(input_vectorized)[0]
    predicted_emotion = lb.inverse_transform([predicted_label])[0]
    label = np.max(lg.predict(input_vectorized))

    return predicted_emotion, label

# App streamlit================================

st.title("Les 6 emotions détecté par la NLP")
st.write(['Joy', 'Fear', 'Love', 'Anger', 'Sadness', 'Surprise'])
input_text = st.text_input("Déposer ici votre commentaire")

if st.button("predire"):
   predicted_emotion, label = predict_emotion(input_text)
   st.write("Emotion predit:", predicted_emotion)
   st.write("Label predit:", label)


