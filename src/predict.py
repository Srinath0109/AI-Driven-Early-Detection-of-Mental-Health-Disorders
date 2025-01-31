import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess import clean_text

# Load model & tokenizer
model = tf.keras.models.load_model("models/mental_health_lstm.h5")
word_index = np.load("models/tokenizer.npy", allow_pickle=True).item()

def predict_text(text):
    """Predicts mental health disorder from input text."""
    cleaned = clean_text(text)
    
    sequence = [[word_index.get(word, 0) for word in cleaned.split()]]
    padded = pad_sequences(sequence, maxlen=100)
    
    prediction = model.predict(padded)[0][0]
    return "⚠️ High Risk of Mental Health Disorder" if prediction > 0.5 else "✅ No Significant Signs Detected"

if __name__ == "__main__":
    text_input = "I've been feeling so alone and hopeless lately."
    print(predict_text(text_input))
