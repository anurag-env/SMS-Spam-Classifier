import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score

from preprocess import transform_text


# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

# Remove unnecessary columns
df.drop(
    columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"],
    inplace=True
)

# Rename columns
df.rename(
    columns={
        "v1": "target",
        "v2": "text"
    },
    inplace=True
)

# Convert ham/spam to 0/1
df["target"] = df["target"].map({
    "ham": 0,
    "spam": 1
})

# Remove duplicates
df.drop_duplicates(
    keep="first",
    inplace=True
)

# Text preprocessing
df["transformed"] = df["text"].apply(transform_text)

# Features and target
X = df["transformed"]
y = df["target"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=2,
    stratify=y
)

# TF-IDF
tfidf = TfidfVectorizer(
    max_features=3000
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Model
mnb = MultinomialNB()

mnb.fit(
    X_train_tfidf,
    y_train
)

# Prediction
y_pred = mnb.predict(X_test_tfidf)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)

# Save model
joblib.dump(mnb, "model.pkl")

# Save TF-IDF vectorizer
joblib.dump(tfidf, "tfidf.pkl")

print("Model saved successfully.")