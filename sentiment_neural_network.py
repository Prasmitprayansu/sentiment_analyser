"""
Sentiment Analysis: Neural Network Model
Week 2 upgrade - builds on baseline

Requirements:
    pip install torch torchvision

Usage:
    python sentiment_neural_network.py

Key improvements over baseline:
    - Non-linear decision boundaries
    - Learned representations vs static TF-IDF
    - Shows you can code deep learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ============================================================================
# NEURAL NETWORK ARCHITECTURE
# ============================================================================

class SentimentNeuralNet(nn.Module):
    """
    Simple but effective neural network for sentiment analysis
    
    Architecture:
        Input (TF-IDF features: 5000) 
        → Dense (256) + ReLU 
        → Dropout (0.3) 
        → Dense (128) + ReLU 
        → Dropout (0.3) 
        → Output (1) + Sigmoid
    """
    
    def __init__(self, input_dim=5000, hidden_dim1=256, hidden_dim2=128, dropout=0.3):
        super(SentimentNeuralNet, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(hidden_dim2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.sigmoid(x)
        
        return x


# ============================================================================
# DATA PREPARATION (Same as baseline)
# ============================================================================

def load_data(filepath):
    """Load sentiment140 dataset"""
    df = pd.read_csv(
        filepath,
        header=None,
        encoding='latin-1',
        names=['target', 'id', 'date', 'flag', 'user', 'text']
    )
    df['target'] = df['target'].map({0: 0, 4: 1})
    return df[['text', 'target']]


def clean_text(text):
    """Text preprocessing"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s\.\!\?]', '', text)
    text = ' '.join(text.split())
    return text


print("Loading and preprocessing data...")
df = pd.read_csv(
    'training.1600000.processed.noemoticon.csv',
    header=None,
    encoding='latin-1',
    names=['target', 'id', 'date', 'flag', 'user', 'text']
)
df['target'] = df['target'].map({0: 0, 4: 1})
df = df[['text', 'target']].sample(n=50000, random_state=42)
df['text'] = df['text'].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['target'],
    test_size=0.2,
    random_state=42,
    stratify=df['target']
)

# Vectorize with TF-IDF
print("Vectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.8
)

X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = vectorizer.transform(X_test).toarray()

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_tfidf).to(DEVICE)
y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1).to(DEVICE)

X_test_tensor = torch.FloatTensor(X_test_tfidf).to(DEVICE)
y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1).to(DEVICE)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Training set: {len(X_train_tensor)} samples")
print(f"Test set: {len(X_test_tensor)} samples")
print(f"Feature dimension: {X_train_tfidf.shape[1]}\n")

# ============================================================================
# TRAINING LOOP
# ============================================================================

print("=" * 60)
print("TRAINING NEURAL NETWORK")
print("=" * 60 + "\n")

model = SentimentNeuralNet(input_dim=X_train_tfidf.shape[1]).to(DEVICE)
criterion = nn.BCELoss()  # Binary cross-entropy for binary classification
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Track losses for visualization
train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    # Training phase
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test_tensor)
        val_loss = criterion(val_outputs, y_test_tensor).item()
        val_losses.append(val_loss)
    
    if (epoch + 1) % 2 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

print("\n✅ Training complete!\n")

# ============================================================================
# EVALUATION
# ============================================================================

print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60 + "\n")

model.eval()
with torch.no_grad():
    train_preds = (model(X_train_tensor) > 0.5).cpu().numpy().flatten()
    test_preds = (model(X_test_tensor) > 0.5).cpu().numpy().flatten()

train_accuracy = accuracy_score(y_train, train_preds)
test_accuracy = accuracy_score(y_test, test_preds)

print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}\n")

print("Test Set Classification Report:")
print(classification_report(y_test, test_preds, target_names=['Negative', 'Positive']))

cm = confusion_matrix(y_test, test_preds)
print(f"\nConfusion Matrix:\n{cm}")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "=" * 60)
print("SAVING VISUALIZATIONS")
print("=" * 60 + "\n")

# Plot training curves
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(train_losses, label='Training Loss', marker='o')
ax.plot(val_losses, label='Validation Loss', marker='s')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Neural Network Training History')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_curves.png', dpi=100, bbox_inches='tight')
print("✅ Saved: training_curves.png")

# Plot confusion matrix
import seaborn as sns
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title('Neural Network Confusion Matrix')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_xticklabels(['Negative', 'Positive'])
ax.set_yticklabels(['Negative', 'Positive'])
plt.tight_layout()
plt.savefig('nn_confusion_matrix.png', dpi=100, bbox_inches='tight')
print("✅ Saved: nn_confusion_matrix.png")

# ============================================================================
# SAVE MODEL
# ============================================================================

torch.save(model.state_dict(), 'neural_network_model.pt')
print("✅ Saved: neural_network_model.pt")

# Save vectorizer for inference
import pickle
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("✅ Saved: vectorizer.pkl")

# ============================================================================
# COMPARISON WITH BASELINE
# ============================================================================

print("\n" + "=" * 60)
print("COMPARISON: BASELINE vs NEURAL NETWORK")
print("=" * 60 + "\n")

results = pd.DataFrame({
    'Model': ['Naive Bayes', 'Logistic Regression', 'Neural Network'],
    'Test Accuracy': [0.7850, 0.8124, round(test_accuracy, 4)]  # Update NB/LR from your baseline
})

print(results.to_string(index=False))
print(f"\nBest Model: {results.loc[results['Test Accuracy'].idxmax(), 'Model']}")

# ============================================================================
# INFERENCE EXAMPLE
# ============================================================================

print("\n" + "=" * 60)
print("INFERENCE EXAMPLE")
print("=" * 60 + "\n")

def predict_sentiment(text, model, vectorizer, device):
    """Predict sentiment of a single tweet"""
    # Clean and vectorize
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned]).toarray()
    
    # Convert to tensor and predict
    features_tensor = torch.FloatTensor(features).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(features_tensor).cpu().numpy()[0][0]
    
    sentiment = 'Positive' if pred > 0.5 else 'Negative'
    confidence = pred if pred > 0.5 else 1 - pred
    
    return sentiment, confidence

# Test with example tweets
test_tweets = [
    "I love this movie! Amazing performance!",
    "Worst experience ever. Total waste of time.",
    "It's okay, nothing special."
]

for tweet in test_tweets:
    sentiment, confidence = predict_sentiment(tweet, model, vectorizer, DEVICE)
    print(f"Tweet: {tweet}")
    print(f"Sentiment: {sentiment} ({confidence:.4f})\n")

print("=" * 60)
print("✨ Neural network training complete!")
print("=" * 60)
