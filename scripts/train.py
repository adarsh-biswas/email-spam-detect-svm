import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load Cleaned Dataset
dataset_path = os.path.join(os.path.dirname(__file__), "../data/cleaned_spam.csv")
df = pd.read_csv(dataset_path)

# Split Data into Features and Labels
X = df["cleaned_text"]
y = df["label"]

# Convert Text into Numerical Vectors (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# Split Dataset (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train SVM Model (Using RBF Kernel)
svm_model = SVC(kernel="rbf", C=1.0, gamma="scale")
svm_model.fit(X_train, y_train)

# Evaluate Model
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Print Model Performance
print(" Model Training Complete!")
print(f" Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# Save Model & Vectorizer
models_dir = os.path.join(os.path.dirname(__file__), "../models/")
os.makedirs(models_dir, exist_ok=True)

model_path = os.path.join(models_dir, "svm_model.pkl")
vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")

with open(model_path, "wb") as model_file:
    pickle.dump(svm_model, model_file)

with open(vectorizer_path, "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print(f" SVM Model saved at: {model_path}")
print(f" TF-IDF Vectorizer saved at: {vectorizer_path}")
