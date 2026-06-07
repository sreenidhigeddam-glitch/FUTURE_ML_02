"""
Support Ticket Classification & Prioritization System
======================================================
Author  : Future Interns ML Task 2 (2026)
Dataset : Kaggle — Customer Support Ticket Dataset
Link    : https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset
"""

import pandas as pd
import numpy as np
import re
import os
import warnings
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# STEP 1 — DOWNLOAD NLTK RESOURCES
# ─────────────────────────────────────────────
def download_nltk():
    for r in ['punkt', 'stopwords', 'wordnet', 'punkt_tab']:
        nltk.download(r, quiet=True)

download_nltk()


# ─────────────────────────────────────────────
# STEP 2 — LOAD & CLEAN KAGGLE DATASET
# ─────────────────────────────────────────────
def load_kaggle_data(csv_path="data/support_tickets.csv"):
    print("📂 Loading Kaggle dataset...")
    df = pd.read_csv(csv_path)

    print(f"\n   Raw columns found: {df.columns.tolist()}")
    print(f"   Raw shape        : {df.shape}")

    # ── Rename Kaggle columns to standard names ──
    df = df.rename(columns={
        "Ticket Description" : "description",
        "Ticket Type"        : "category",
        "Ticket Priority"    : "priority"
    })

    # Keep only the 3 columns we need
    df = df[["description", "category", "priority"]].dropna()

    # Clean up any leading/trailing spaces in labels
    df["category"] = df["category"].str.strip()
    df["priority"]  = df["priority"].str.strip()

    print(f"\n✅ Dataset ready!")
    print(f"   Total tickets  : {len(df)}")
    print(f"   Categories     : {df['category'].unique().tolist()}")
    print(f"   Priority levels: {df['priority'].unique().tolist()}")

    return df


# ─────────────────────────────────────────────
# STEP 3 — TEXT PREPROCESSOR
# ─────────────────────────────────────────────
class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def clean(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)     # remove URLs
        text = re.sub(r"[^a-z\s]", " ", text)          # remove special chars & numbers
        text = re.sub(r"\s+", " ", text).strip()        # normalize spaces

        tokens = word_tokenize(text)
        tokens = [
            self.lemmatizer.lemmatize(t)
            for t in tokens
            if t not in self.stop_words and len(t) > 2
        ]
        return " ".join(tokens)

    def transform(self, series):
        return series.apply(self.clean)


# ─────────────────────────────────────────────
# STEP 4 — CLASSIFIER
# ─────────────────────────────────────────────
class TicketClassifier:
    def __init__(self):
        self.preprocessor = TextPreprocessor()

        # Category model — Logistic Regression
        self.category_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=10000,
                sublinear_tf=True
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                C=1.0,
                solver="lbfgs",
                multi_class="multinomial"
            ))
        ])

        # Priority model — Random Forest
        self.priority_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=10000,
                sublinear_tf=True
            )),
            ("clf", RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            ))
        ])

    def train(self, df):
        print("\n🧹 Preprocessing text...")
        df = df.copy()
        df["clean_text"] = self.preprocessor.transform(df["description"])

        X = df["clean_text"]
        y_cat = df["category"]
        y_pri = df["priority"]

        X_train, X_test, y_cat_train, y_cat_test, y_pri_train, y_pri_test = train_test_split(
            X, y_cat, y_pri,
            test_size=0.2,
            random_state=42,
            stratify=y_cat
        )

        print(f"   Train size : {len(X_train)}")
        print(f"   Test size  : {len(X_test)}")

        print("\n🚀 Training Category Classifier (Logistic Regression)...")
        self.category_pipeline.fit(X_train, y_cat_train)

        print("🚀 Training Priority Classifier (Random Forest)...")
        self.priority_pipeline.fit(X_train, y_pri_train)

        print("\n✅ Training complete!")
        return X_test, y_cat_test, y_pri_test

    def evaluate(self, X_test, y_cat_test, y_pri_test):
        print("\n" + "=" * 65)
        print("  📊 EVALUATION RESULTS")
        print("=" * 65)

        cat_preds = self.category_pipeline.predict(X_test)
        pri_preds = self.priority_pipeline.predict(X_test)

        print("\n📌 CATEGORY CLASSIFICATION REPORT")
        print("-" * 65)
        print(classification_report(y_cat_test, cat_preds))

        print("\n📌 PRIORITY CLASSIFICATION REPORT")
        print("-" * 65)
        print(classification_report(y_pri_test, pri_preds))

        return cat_preds, pri_preds

    def predict(self, text):
        """Predict category and priority for a single new ticket."""
        clean = self.preprocessor.clean(text)

        category      = self.category_pipeline.predict([clean])[0]
        priority       = self.priority_pipeline.predict([clean])[0]
        cat_proba      = self.category_pipeline.predict_proba([clean])[0]
        pri_proba      = self.priority_pipeline.predict_proba([clean])[0]
        cat_classes    = self.category_pipeline.classes_
        pri_classes    = self.priority_pipeline.classes_

        return {
            "input_text"          : text,
            "predicted_category"  : category,
            "predicted_priority"  : priority,
            "category_confidence" : {c: round(p * 100, 1) for c, p in zip(cat_classes, cat_proba)},
            "priority_confidence" : {c: round(p * 100, 1) for c, p in zip(pri_classes, pri_proba)},
        }

    def save(self, path="models"):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.category_pipeline, f"{path}/category_model.pkl")
        joblib.dump(self.priority_pipeline, f"{path}/priority_model.pkl")
        print(f"\n💾 Models saved → '{path}/' folder")

    def load(self, path="models"):
        self.category_pipeline = joblib.load(f"{path}/category_model.pkl")
        self.priority_pipeline = joblib.load(f"{path}/priority_model.pkl")
        print("✅ Models loaded successfully.")


# ─────────────────────────────────────────────
# STEP 5 — VISUALIZATIONS
# ─────────────────────────────────────────────
def plot_distribution(df):
    os.makedirs("outputs", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Category
    cat_counts = df["category"].value_counts()
    axes[0].barh(cat_counts.index, cat_counts.values, color="#4C72B0")
    axes[0].set_title("Ticket Category Distribution", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Number of Tickets")
    for i, v in enumerate(cat_counts.values):
        axes[0].text(v + 10, i, str(v), va="center", fontsize=9)

    # Priority
    pri_counts = df["priority"].value_counts()
    color_map = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#2ecc71",
                 "Critical": "#8e44ad", "Urgent": "#c0392b"}
    bar_colors = [color_map.get(p, "#95a5a6") for p in pri_counts.index]
    axes[1].bar(pri_counts.index, pri_counts.values, color=bar_colors, edgecolor="white")
    axes[1].set_title("Ticket Priority Distribution", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Number of Tickets")
    for i, v in enumerate(pri_counts.values):
        axes[1].text(i, v + 5, str(v), ha="center", fontsize=9)

    plt.suptitle("Kaggle Customer Support Ticket Dataset — Overview",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("outputs/data_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("📊 Saved → outputs/data_distribution.png")


def plot_confusion_matrices(y_cat_test, cat_preds, y_pri_test, pri_preds):
    os.makedirs("outputs", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Category confusion matrix
    labels_cat = sorted(y_cat_test.unique())
    cm_cat = confusion_matrix(y_cat_test, cat_preds, labels=labels_cat)
    sns.heatmap(cm_cat, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels_cat, yticklabels=labels_cat, ax=axes[0])
    axes[0].set_title("Category Classifier — Confusion Matrix", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Predicted", fontsize=11)
    axes[0].set_ylabel("Actual", fontsize=11)
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].tick_params(axis="y", rotation=0)

    # Priority confusion matrix
    labels_pri = sorted(y_pri_test.unique())
    cm_pri = confusion_matrix(y_pri_test, pri_preds, labels=labels_pri)
    sns.heatmap(cm_pri, annot=True, fmt="d", cmap="Oranges",
                xticklabels=labels_pri, yticklabels=labels_pri, ax=axes[1])
    axes[1].set_title("Priority Classifier — Confusion Matrix", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Predicted", fontsize=11)
    axes[1].set_ylabel("Actual", fontsize=11)

    plt.tight_layout()
    plt.savefig("outputs/confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("📊 Saved → outputs/confusion_matrices.png")


# ─────────────────────────────────────────────
# STEP 6 — MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 65)
    print("   SUPPORT TICKET CLASSIFICATION & PRIORITIZATION SYSTEM")
    print("   Kaggle Dataset — Future Interns ML Task 2 (2026)")
    print("=" * 65)

    # Load data
    df = load_kaggle_data("data/support_tickets.csv")

    # Plot distribution
    print("\n📊 Plotting data distribution...")
    plot_distribution(df)

    # Train
    classifier = TicketClassifier()
    X_test, y_cat_test, y_pri_test = classifier.train(df)

    # Evaluate
    cat_preds, pri_preds = classifier.evaluate(X_test, y_cat_test, y_pri_test)

    # Confusion matrices
    print("\n📊 Plotting confusion matrices...")
    plot_confusion_matrices(y_cat_test, cat_preds, y_pri_test, pri_preds)

    # Save models
    classifier.save()

    # Demo predictions
    print("\n" + "=" * 65)
    print("  🔍 DEMO — PREDICTING ON NEW TICKETS")
    print("=" * 65)

    test_tickets = [
        "I have been charged twice for my subscription this month. Please refund immediately.",
        "The app crashes every time I try to log in. I cannot access my account at all.",
        "I would like to know how to upgrade my plan to the premium version.",
        "My password reset email never arrived. I have been locked out for 2 days.",
        "Can I get a refund for my last payment? I cancelled within the refund window.",
    ]

    for ticket in test_tickets:
        result = classifier.predict(ticket)
        print(f"\n📝 {result['input_text']}")
        print(f"   ➤ Category : {result['predicted_category']}")
        print(f"   ➤ Priority  : {result['predicted_priority']}")
        best_cat = max(result['category_confidence'], key=result['category_confidence'].get)
        print(f"   ➤ Confidence: {result['category_confidence'][best_cat]}%")

    print("\n✅ All done! Check the 'outputs/' folder for saved charts.")


if __name__ == "__main__":
    main()
