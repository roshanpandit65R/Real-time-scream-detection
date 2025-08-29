# utils.py
import librosa
import numpy as np
import geocoder
import requests
from keras.models import load_model

# Load trained model
model = load_model("model/scream_model.h5")

# Telegram Bot credentials
BOT_TOKEN = "pls paste your Bot Token "
CHAT_ID = "pls paste your chat Id "  # ← Replace this below using instructions

def extract_features_from_live(audio):
    mfccs = librosa.feature.mfcc(y=audio, sr=44100, n_mfcc=40)
    return np.mean(mfccs.T, axis=0).reshape(1, -1)

def get_location():
    g = geocoder.ip('me')
    return g.latlng or ["Unknown", "Unknown"]

def send_alert(location):
    message = f"🚨 Scream Detected!\n\n📍 Location: {location[0]}, {location[1]}\nPlease take immediate action."
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": message
    }
    try:
        response = requests.post(url, data=data)
        print("✅ Alert sent via Telegram")
    except Exception as e:
        print("❌ Failed to send Telegram alert:", e)

def predict_scream(audio):
    features = extract_features_from_live(audio)
    prediction = model.predict(features)
    return np.argmax(prediction), prediction[0]
