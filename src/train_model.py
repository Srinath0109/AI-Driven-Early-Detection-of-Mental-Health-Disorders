import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from preprocess import clean_text

# Load dataset
df = pd.read_csv("data/mental_health_dataset.csv")  # Dataset with "text" and "label"

# Preprocess text
df["clean_text"] = df["text"].apply(clean_text)

# Tokenize
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df["clean_text"])
sequences = tokenizer.texts_to_sequences(df["clean_text"])
X = pad_sequences(sequences, maxlen=100)
y = df["label"].values

# Build Model
model = Sequential([
    Embedding(5000, 128, input_length=100),
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2)

# Save model
model.save("models/mental_health_lstm.h5")
np.save("models/tokenizer.npy", tokenizer.word_index)
print("✅ Model trained and saved!")
