# ================================================================
# 🧠 Consumer Complaint Text Classification
# Categories:
#   0 - Credit reporting, repair, or other
#   1 - Debt collection
#   2 - Consumer Loan
#   3 - Mortgage
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import string
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

csv_path = r"C:\Users\Parameshwar B\OneDrive\Desktop\newfollder\complaints_copy.csv"  # Change as needed
df = pd.read_csv(csv_path)

print("✅ Dataset Loaded Successfully!")
print("Shape:", df.shape)
print(df.head())


df = df[['Product', 'Consumer complaint narrative']].dropna()


category_map = {
    'Credit reporting, repair, or other': 0,
    'Debt collection': 1,
    'Consumer Loan': 2,
    'Mortgage': 3
}

df = df[df['Product'].isin(category_map.keys())]
df['label'] = df['Product'].map(category_map)

print("\n📊 Category Distribution:")
print(df['label'].value_counts())


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['clean_text'] = df['Consumer complaint narrative'].apply(clean_text)

print("\n🧹 Sample Cleaned Text:")
print(df['clean_text'].head())

X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)
print("\n✅ Data split completed!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=300),
    "Linear SVM": LinearSVC()
}

results = {}
model_dir = r"C:\Users\Parameshwar B\OneDrive\Desktop\newfollder\models"
os.makedirs(model_dir, exist_ok=True)

for name, model in models.items():
    print(f"\n🔹 Training {name}...")
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"✅ Accuracy for {name}: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    
    # Save each trained model
    model_path = os.path.join(model_dir, f"{name.replace(' ', '_').lower()}_model.pkl")
    joblib.dump(model, model_path)
    print(f"💾 Saved model to: {model_path}")

# Save TF-IDF vectorizer
vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
joblib.dump(vectorizer, vectorizer_path)
print(f"💾 Saved vectorizer to: {vectorizer_path}")


plt.figure(figsize=(7,4))
sns.barplot(x=list(results.keys()), y=list(results.values()), palette="viridis")
plt.title("Model Performance Comparison")
plt.ylabel("Accuracy")
plt.xlabel("Models")
plt.show()


best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\n🏆 Best Performing Model: {best_model_name}")

# Confusion Matrix
y_pred_best = best_model.predict(X_test_tfidf)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


sample_texts = [
    "I was charged extra fees on my mortgage loan.",
    "They reported incorrect information on my credit report.",
    "The debt collector was rude and harassed me.",
    "My consumer loan application was rejected unfairly."
]

sample_tfidf = vectorizer.transform(sample_texts)
pred_labels = best_model.predict(sample_tfidf)

reverse_map = {v: k for k, v in category_map.items()}

print("\n🔮 Sample Predictions:")
for text, label in zip(sample_texts, pred_labels):
    print(f"Text: {text}\n→ Predicted Category: {reverse_map[label]}\n")


best_model_path = os.path.join(model_dir, f"best_model_{best_model_name.replace(' ', '_').lower()}.pkl")
joblib.dump(best_model, best_model_path)
print(f"🏁 Best model saved at: {best_model_path}")
