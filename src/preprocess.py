import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("stopwords")
nltk.download("punkt")

def clean_text(text):
    """Removes URLs, special characters, and stopwords."""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r"@\w+|\#", "", text)  # Remove mentions/hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    text = text.lower()  # Convert to lowercase

    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)
    return " ".join([word for word in words if word not in stop_words])

if __name__ == "__main__":
    sample_text = "Feeling really down today... Nothing seems to help. 😞 #depressed"
    print("Cleaned:", clean_text(sample_text))
