# main.py
import sounddevice as sd
import numpy as np
import time
from utils import predict_scream, get_location, send_alert

DURATION = 3  # seconds
THRESHOLD = 0.90

print("🟢 Real-time Scream Detection Started...")

while True:
    recording = sd.rec(int(DURATION * 44100), samplerate=44100, channels=1)
    sd.wait()
    audio_data = recording.flatten()
    label, prob = predict_scream(audio_data)
    
    if label == 1 and prob[1] > THRESHOLD:
        print("🚨 Scream Detected! Confidence:", prob[1])
        location = get_location()
        send_alert(location)
    else:
        print("🙂 Normal sound detected.")
    
    time.sleep(1)
