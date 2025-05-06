import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize Lemmatizer & Stopwords
lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))

# Define Text Cleaning Function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = text.split()  # Tokenize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatization & stopword removal
    return ' '.join(words)

# Load Dataset
dataset_path = os.path.join(os.path.dirname(__file__), "../data/spam.csv")
df = pd.read_csv(dataset_path, encoding="latin-1")

# Rename columns properly
df.columns = ["text", "label"]

# Ensure 'label' is numerical
df['label'] = df['label'].astype(int)

# Apply text cleaning
df['cleaned_text'] = df['text'].apply(clean_text)

# Print dataset insights
print(" Data Loaded Successfully!")
print(df['label'].value_counts())

# Save Cleaned Data
cleaned_path = os.path.join(os.path.dirname(__file__), "../data/cleaned_spam.csv")
df[['cleaned_text', 'label']].to_csv(cleaned_path, index=False)

print(f"Cleaned dataset saved at: {cleaned_path}")
