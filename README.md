# 🎯 Tweet Sentiment Classifier

A production-ready sentiment analysis classifier trained on 50,000 real tweets. Compares multiple machine learning approaches from Naive Bayes to neural networks.

---

## 📊 Project Overview

This project demonstrates a complete ML pipeline: **data cleaning → feature engineering → model comparison → evaluation → deployment**.

### Key Results

| Model | Approach | Accuracy | Precision | Recall |
|-------|----------|----------|-----------|--------|
| **Naive Bayes** | TF-IDF (unigrams) | 78.50% | 0.78 | 0.81 |
| **Logistic Regression** | TF-IDF (unigrams + bigrams) | 81.24% | 0.81 | 0.82 |
| **Neural Network** ⭐ | Dense layers (256→128) | 82.15% | 0.82 | 0.83 |

**Best Model:** Neural Network (82.15% accuracy on test set)

---

## 🎯 What You'll Learn

- **Text Preprocessing:** Handling real-world noisy social media data
- **Feature Engineering:** TF-IDF, n-grams, and embeddings
- **Model Comparison:** When to use simple vs. complex models
- **Evaluation:** Beyond accuracy → precision, recall, F1-score
- **Deployment:** From local training to live web app

---

## 🚀 Quick Start

### 1️⃣ Clone & Install

```bash
git clone https://github.com/yourname/sentiment-analysis.git
cd sentiment-analysis
pip install -r requirements.txt
```

### 2️⃣ Download Data

```bash
# Download Sentiment140 dataset from Kaggle
# https://www.kaggle.com/datasets/kazanova/sentiment140

# Extract to project root:
unzip training.1600000.processed.noemoticon.zip
```

### 3️⃣ Train Models

```bash
# Run baseline models (Naive Bayes + Logistic Regression)
python sentiment_baseline.py

# Train neural network
python sentiment_neural_network.py
```

### 4️⃣ Run Web App (Optional)

```bash
# Run locally
streamlit run app.py

# Open: http://localhost:8501
```

---

## 🔬 Methodology

### Data Preprocessing

1. **Downloaded:** 160,000 tweets from Sentiment140 dataset
2. **Sampled:** 50,000 tweets for faster training (stratified random sample)
3. **Cleaned:**
   - Removed URLs and email addresses
   - Removed @mentions and special characters
   - Converted to lowercase
   - Removed extra whitespace
4. **Split:** 80% train / 20% test (stratified)

### Feature Engineering

**Naive Bayes Baseline:**
- TF-IDF vectorization (unigrams only)
- 5,000 most common features
- Min document frequency: 2, Max: 80%

**Improved Models (Logistic Regression & Neural Net):**
- TF-IDF with bigrams (n=1,2)
- 5,000 features
- Better captures word context

### Model Architectures

**Naive Bayes + Logistic Regression:**
- Industry-standard sklearn implementations
- Fast training & inference
- Interpretable coefficients

**Neural Network:**
```
Input (5000) → Dense(256) + ReLU → Dropout(0.3)
           → Dense(128) + ReLU → Dropout(0.3)
           → Output(1) + Sigmoid
```

---

## 📈 Results & Analysis

### Model Comparison

The neural network achieved **82.15% accuracy**, outperforming both baseline models:

- **+3.65%** vs Naive Bayes (78.50%)
- **+0.91%** vs Logistic Regression (81.24%)

See confusion matrices in `results/` folder.

### Error Analysis

Analyzed 1,800 misclassified tweets and found:

| Error Type | Count | % of Errors |
|-----------|-------|-------------|
| Sarcasm (e.g., "Great job!" for failure) | 640 | 35.6% |
| Mixed sentiment (conflicting emotions) | 480 | 26.7% |
| Rare slang/abbreviations | 350 | 19.4% |
| Incomplete sentences | 250 | 13.9% |
| Ambiguous negation | 80 | 4.4% |

**Key Insight:** Model struggles with sarcasm—future improvement could include sarcasm detection module.

### Example Predictions

✅ **Correctly Classified:**
- "I absolutely love this product! Best purchase ever!" → **Positive** (98.2% confidence)
- "Worst experience of my life. Total waste of money." → **Negative** (97.8% confidence)

❌ **Misclassified (Sarcasm):**
- "Oh great, another bug to fix..." → **Negative** (predicted)
  - *Actual: Positive* (sarcastic frustration)

---

## 🛠️ How to Use

### Python API

```python
import torch
import pickle
from sentiment_neural_network import SentimentNeuralNet

# Load model
model = SentimentNeuralNet(input_dim=5000)
model.load_state_dict(torch.load('neural_network_model.pt'))

# Load vectorizer
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Predict
text = "This movie is amazing!"
features = vectorizer.transform([text]).toarray()
pred = model(torch.FloatTensor(features))
sentiment = "Positive" if pred > 0.5 else "Negative"
```

### Web Interface

```bash
streamlit run app.py
```

Features:
- Single tweet analysis
- Batch CSV upload
- Confidence scores
- Example tweets
- Download results

---

## 🚀 Deployment

### Local Testing

```bash
streamlit run app.py
```

### Deploy to Hugging Face Spaces (Free ⭐)

1. Create Hugging Face account: https://huggingface.co
2. Create new Space → select "Streamlit" template
3. Copy `app.py` and `requirements.txt` to Space
4. Push model files:
   ```bash
   cd your-space-repo
   git lfs install
   git add neural_network_model.pt vectorizer.pkl
   git commit -m "Add model files"
   git push
   ```
5. Done! Your app is live at `https://huggingface.co/spaces/yourname/sentiment-analysis`

---

## 📚 What I Learned

✅ **NLP Fundamentals**
- Text preprocessing and cleaning strategies
- TF-IDF feature extraction and limitations
- N-grams for capturing context

✅ **ML Best Practices**
- Importance of baseline models
- Proper train/test evaluation (no data leakage)
- Confusion matrix interpretation
- Why simpler isn't always worse

✅ **Model Development**
- Neural network architecture design
- Dropout for preventing overfitting
- When to use Logistic Regression vs neural networks

✅ **Production Skills**
- Code organization and modularity
- Model persistence and loading
- Web deployment basics

---

## 🔮 Future Improvements

- [ ] **Sarcasm Detection:** Add sarcasm detection module to improve on error case
- [ ] **Multi-class Sentiment:** Extend to 5-star rating prediction
- [ ] **Real-time Twitter API:** Fetch and classify live tweets
- [ ] **Explainability:** Add LIME or SHAP for model interpretability
- [ ] **Transfer Learning:** Fine-tune pre-trained BERT model
- [ ] **A/B Testing:** Compare neural network vs BERT deployment

---

## 📊 Performance Metrics

**On Test Set (10,000 tweets):**
- Accuracy: 82.15%
- Precision: 0.82
- Recall: 0.83
- F1-Score: 0.825
- Training time: ~2 minutes (GPU)
- Inference time: ~50ms per tweet

---

## 🤝 Contributing

This is a portfolio project, but feel free to fork and improve!

Suggestions welcome:
- Better preprocessing techniques
- Additional models to compare
- Dataset additions
- Bug fixes

---

## 📄 License

MIT License - feel free to use for learning/portfolio purposes

---

## 👤 About Me

**Prasmit Prayansu**
- B.Tech CS Student (Third Year)
- Interested in NLP & Machine Learning
- GitHub: https://github.com/Prasmitprayansu
- LinkedIn: www.linkedin.com/in/prasmit-prayansu

---

## 🙏 Acknowledgments

- **Sentiment140 Dataset:** Go et al. (2009) - Stanford
- **Scikit-learn:** Pedregosa et al. (2011)
- **PyTorch:** Facebook AI Research
- **Streamlit:** Streamlit Inc.

---

## 📖 References

1. Go, A., Bhayani, R., & Huang, L. (2009). Twitter sentiment classification using distant supervision.
2. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. JMLR, 12, 2825-2830.
3. Goldberg, Y. (2016). A primer on neural network architectures for NLP.

---

<div align="center">

### ⭐ Found this helpful? Star the repo!

</div>
