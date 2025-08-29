# train_model.py
import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.models import load_model

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print("Error extracting features:", e)
        return None

def load_data():
    X, y = [], []
    for label, category in enumerate(['negative', 'positive']):
        folder = f'data/{category}'
        for file_name in os.listdir(folder):
            file_path = os.path.join(folder, file_name)
            features = extract_features(file_path)
            if features is not None:
                X.append(features)
                y.append(label)
    return np.array(X), to_categorical(y)

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential([
    Dense(256, activation='relu', input_shape=(40,)),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))
model.save("model/scream_model.h5")
print("Model saved to model/scream_model.h5")
