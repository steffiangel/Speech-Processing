import streamlit as st
import speech_recognition as sr
from transformers import pipeline
import random

# Load Sentiment Analysis Model
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis")

sentiment_model = load_sentiment_model()

# Function to Transcribe Speech to Text
def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return "Audio not clear, please try again."

# Function to Analyze Sentiment
def analyze_sentiment(text):
    return sentiment_model(text)

# Function to Provide Mood-Enhancing Messages
def get_mood_message(sentiment_label):
    positive_messages = [
        "Keep shining bright! 😊",
        "You're doing amazing, keep it up!",
        "Great vibes! Spread the positivity!"
    ]
    negative_messages = [
        "Don't worry, things will get better. 🌈",
        "Here's a little motivation: *Believe in yourself!* 💪",
        "Why don’t scientists trust atoms? Because they make up everything! 😄"
    ]
    neutral_messages = [
        "Stay balanced and keep going. 🌟",
        "A calm mind brings inner strength and self-confidence.",
        "Enjoy the little things in life. 🌼"
    ]
    
    if sentiment_label == "POSITIVE":
        return random.choice(positive_messages)
    elif sentiment_label == "NEGATIVE":
        return random.choice(negative_messages)
    else:
        return random.choice(neutral_messages)

# Streamlit UI
st.title("Mood Enhancer through Speech Sentiment Analysis")
st.write("Upload an audio file to analyze its sentiment and receive a personalized mood message!")

# Upload Audio
audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac"])
if audio_file:
    st.audio(audio_file, format="audio/wav")

    # Transcription
    st.subheader("Transcription")
    transcript = transcribe_audio(audio_file)
    st.write(f"**Transcript**: {transcript}")

    # Sentiment Analysis
    st.subheader("Sentiment Analysis")
    if transcript:
        sentiment = analyze_sentiment(transcript)
        sentiment_label = sentiment[0]['label']
        sentiment_score = sentiment[0]['score']
        st.write(f"**Sentiment**: {sentiment_label}")
        st.write(f"**Confidence**: {sentiment_score:.2f}")

        # Mood-Enhancing Message
        st.subheader("Mood-Enhancing Message")
        mood_message = get_mood_message(sentiment_label)
        st.success(mood_message)
