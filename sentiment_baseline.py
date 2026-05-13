"""
Sentiment Analysis Baseline Model
Start here: Day 1-3 of your project

Usage:
    python sentiment_baseline.py

Output:
    - Saves trained model
    - Prints evaluation metrics
    - Creates confusion matrix plot
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: DATA LOADING & CLEANING
# ============================================================================

def load_data(filepath):
    """Load sentiment140 dataset from CSV"""
    # Columns: (target, ids, date, flag, user, text)
    df = pd.read_csv(
        filepath,
        header=None,
        encoding='latin-1',
        names=['target', 'id', 'date', 'flag', 'user', 'text']
    )
    
    # Convert labels: 4 (positive) → 1, 0 (negative) → 0
    df['target'] = df['target'].map({0: 0, 4: 1})
    
    return df[['text', 'target']]


def clean_text(text):
    """Basic text preprocessing"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove mentions and hashtags (keep the words)
    text = re.sub(r'@\w+', '', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s\.\!\?]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


# ============================================================================
# STEP 2: LOAD & PREPARE DATA
# ============================================================================

print("Loading data...")
# Download from: https://www.kaggle.com/datasets/kazanova/sentiment140
df = load_data('training.1600000.processed.noemoticon.csv')

# Use a sample for faster training (remove this for final submission)
df = df.sample(n=50000, random_state=42)

print(f"Dataset shape: {df.shape}")
print(f"Class distribution:\n{df['target'].value_counts()}\n")

# Clean text
print("Cleaning text...")
df['text'] = df['text'].apply(clean_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['target'],
    test_size=0.2,
    random_state=42,
    stratify=df['target']
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}\n")


# ============================================================================
# STEP 3: BASELINE MODEL (NAIVE BAYES)
# ============================================================================

print("=" * 60)
print("MODEL 1: NAIVE BAYES + TF-IDF")
print("=" * 60)

# Vectorize text using TF-IDF
vectorizer_nb = TfidfVectorizer(
    max_features=5000,  # Limit features for speed
    ngram_range=(1, 1),  # Unigrams only
    min_df=2,  # Ignore very rare words
    max_df=0.8  # Ignore very common words
)

X_train_tfidf = vectorizer_nb.fit_transform(X_train)
X_test_tfidf = vectorizer_nb.transform(X_test)

# Train Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred_nb = nb_model.predict(X_test_tfidf)
nb_accuracy = accuracy_score(y_test, y_pred_nb)

print(f"\nAccuracy: {nb_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_nb, target_names=['Negative', 'Positive']))

cm_nb = confusion_matrix(y_test, y_pred_nb)
print(f"\nConfusion Matrix:\n{cm_nb}")


# ============================================================================
# STEP 4: IMPROVED MODEL (LOGISTIC REGRESSION)
# ============================================================================

print("\n" + "=" * 60)
print("MODEL 2: LOGISTIC REGRESSION + TF-IDF (with bigrams)")
print("=" * 60)

# Vectorize with bigrams for better context
vectorizer_lr = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),  # Unigrams + bigrams
    min_df=2,
    max_df=0.8
)

X_train_tfidf_lr = vectorizer_lr.fit_transform(X_train)
X_test_tfidf_lr = vectorizer_lr.transform(X_test)

# Train Logistic Regression
lr_model = LogisticRegression(max_iter=200, random_state=42)
lr_model.fit(X_train_tfidf_lr, y_train)

# Evaluate
y_pred_lr = lr_model.predict(X_test_tfidf_lr)
lr_accuracy = accuracy_score(y_test, y_pred_lr)

print(f"\nAccuracy: {lr_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr, target_names=['Negative', 'Positive']))

cm_lr = confusion_matrix(y_test, y_pred_lr)
print(f"\nConfusion Matrix:\n{cm_lr}")


# ============================================================================
# STEP 5: COMPARISON & VISUALIZATION
# ============================================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

comparison = pd.DataFrame({
    'Model': ['Naive Bayes', 'Logistic Regression'],
    'Accuracy': [nb_accuracy, lr_accuracy]
})
print(f"\n{comparison.to_string(index=False)}\n")

# Plot confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Naive Bayes Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_title('Logistic Regression Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=100, bbox_inches='tight')
print("✅ Saved: confusion_matrices.png")

# Plot accuracy comparison
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(comparison['Model'], comparison['Accuracy'], color=['skyblue', 'lightcoral'])
ax.set_ylabel('Accuracy')
ax.set_title('Model Comparison: Accuracy')
ax.set_ylim(0.7, 0.95)
for i, v in enumerate(comparison['Accuracy']):
    ax.text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=100, bbox_inches='tight')
print("✅ Saved: model_comparison.png")

# Save results to CSV
comparison.to_csv('results.csv', index=False)
print("✅ Saved: results.csv")


# ============================================================================
# STEP 6: ERROR ANALYSIS (Optional but Impressive)
# ============================================================================

print("\n" + "=" * 60)
print("ERROR ANALYSIS: Wrong Predictions from Logistic Regression")
print("=" * 60)

wrong_indices = np.where(y_pred_lr != y_test)[0]
print(f"\nTotal mistakes: {len(wrong_indices)} / {len(y_test)} ({100*len(wrong_indices)/len(y_test):.2f}%)\n")

# Show 5 random mistakes
print("Sample mistakes:")
for i, idx in enumerate(np.random.choice(wrong_indices, min(5, len(wrong_indices)), replace=False)):
    actual = 'Positive' if y_test.iloc[idx] == 1 else 'Negative'
    predicted = 'Positive' if y_pred_lr[idx] == 1 else 'Negative'
    text = X_test.iloc[idx]
    print(f"\n{i+1}. Actual: {actual}, Predicted: {predicted}")
    print(f"   Tweet: {text[:100]}...")


# ============================================================================
# STEP 7: SAVE MODELS
# ============================================================================

import pickle

with open('nb_model.pkl', 'wb') as f:
    pickle.dump((nb_model, vectorizer_nb), f)
print("\n✅ Saved: nb_model.pkl")

with open('lr_model.pkl', 'wb') as f:
    pickle.dump((lr_model, vectorizer_lr), f)
print("✅ Saved: lr_model.pkl")

print("\n" + "=" * 60)
print("🎉 BASELINE COMPLETE!")
print("=" * 60)
print("\nNext steps:")
print("1. Review confusion_matrices.png and model_comparison.png")
print("2. Analyze errors in the 'ERROR ANALYSIS' section above")
print("3. Build neural network model in next script")
print("4. Push to GitHub with clear commit message")
