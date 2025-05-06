import os
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources are available
nltk.download("stopwords")
nltk.download("wordnet")

# Load Pre-trained Model & Vectorizer
models_dir = os.path.join(os.path.dirname(__file__), "../models/")
model_path = os.path.join(models_dir, "svm_model.pkl")
vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")

with open(model_path, "rb") as model_file:
    svm_model = pickle.load(model_file)

with open(vectorizer_path, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)


# Preprocessing Function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    words = text.split()

    # Remove stopwords and apply lemmatization
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return " ".join(words)


# Prediction Function
def predict_email(email_text):
    processed_text = preprocess_text(email_text)
    vectorized_text = vectorizer.transform([processed_text])

    prediction = svm_model.predict(vectorized_text)

    return "Spam" if prediction[0] == 1 else "Not Spam"


# User Input for Email
if __name__ == "__main__":
    print("Spam Email Detection System")
    while True:
        user_email = input("\nEnter an email text (or type 'exit' to quit): ")
        if user_email.lower() == "exit":
            print("Exiting...")
            break
        result = predict_email(user_email)
        print(f"Prediction: {result}")
