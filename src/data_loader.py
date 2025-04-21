import os
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Define preprocessing tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# File paths
RAW_TRAIN_PATH = "data/raw/train.csv"
RAW_TEST_PATH = "data/raw/test.csv"
OUT_TRAIN_PATH = "data/processed/processed_train.parquet"
OUT_TEST_PATH = "data/processed/processed_test.parquet"

# Function to preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered = [w for w in tokens if w.isalpha() and w not in stop_words]
    lemmatized = [lemmatizer.lemmatize(w) for w in filtered]
    return " ".join(lemmatized)

# Function to clean, encode, and save
def clean_and_save(input_path, output_path):
    df = pd.read_csv(input_path)

    if "review" not in df.columns or "sentiment" not in df.columns:
        raise ValueError(f"{input_path} must contain 'review' and 'sentiment' columns")

    print(f"Loaded {len(df)} rows from {input_path}")

    # Drop duplicates
    df = df.drop_duplicates(subset="review")
    print(f"Remaining after duplicate removal: {len(df)} rows")

    # Encode sentiment
    df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})
    if df["sentiment"].isnull().any():
        raise ValueError("Found unexpected values in sentiment column after encoding.")

    # Preprocess review text
    print(f"Preprocessing text from {input_path}...")
    df["cleaned"] = df["review"].astype(str).apply(preprocess_text)

    # Final structure
    df = df[["review", "sentiment", "cleaned"]]

    # Save as parquet
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")

def main():
    clean_and_save(RAW_TRAIN_PATH, OUT_TRAIN_PATH)
    clean_and_save(RAW_TEST_PATH, OUT_TEST_PATH)
    print("Data preprocessing complete.")

if __name__ == "__main__":
    main()