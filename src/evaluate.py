import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
from preprocess import clean_text
from train_model import tokenizer

# Load dataset
df = pd.read_csv("data/test_data.csv")
df["clean_text"] = df["text"].apply(clean_text)

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(df["clean_text"])
X_test = pad_sequences(sequences, maxlen=100)
y_test = df["label"].values

# Load model
model = tf.keras.models.load_model("models/mental_health_lstm.h5")

# Predict
preds = (model.predict(X_test) > 0.5).astype(int)
print("✅ Accuracy:", accuracy_score(y_test, preds))
print("📊 Classification Report:\n", classification_report(y_test, preds))
