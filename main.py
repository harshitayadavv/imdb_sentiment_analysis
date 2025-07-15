import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load word index and reverse mapping
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the trained model
model = load_model('simple_rnn_imdb.h5')

# Optional: Decode numerical review back to words
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Streamlit App UI
st.set_page_config(page_title="IMDB Sentiment Classifier", layout="centered")
st.title('🎬 IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review and the model will predict whether it is Positive or Negative.')

# User input box
user_input = st.text_area('✍️ Movie Review:')

# Predict button
if st.button('🔍 Classify Sentiment'):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review before classifying.")
    else:
        preprocessed_input = preprocess_text(user_input)
        prediction = model.predict(preprocessed_input)
        sentiment = 'Positive 😊' if prediction[0][0] > 0.5 else 'Negative 😞'
        st.markdown(f"### 🎯 Sentiment: **{sentiment}**")
