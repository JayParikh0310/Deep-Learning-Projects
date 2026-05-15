import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}


model = load_model('simplernn.keras')

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = []

    for word in words:
        index = word_index.get(word, 2) + 3

        if index >= 10000:
            index = 2   # unknown word token

        encoded_review.append(index)

    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)

    prediction=model.predict(preprocessed_input)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    
    return sentiment, prediction[0][0]



st.title('IMDB Movie Review Sentiment Anaysis')
st.write('Enter a movie review to classify it as possitive or negative.')

user_input = st.text_area('Movie Review')

if st.button('Classify'):
    sentiment,score=predict_sentiment(user_input)

    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {score}')
else:
    st.write('Please Write a movie review.')