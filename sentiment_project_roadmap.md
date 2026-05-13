# Sentiment Analysis Portfolio Project: 2-3 Week Sprint

## 🎯 Your Goal
Build a **real-world sentiment classifier** that shows you understand:
- Data preprocessing for NLP
- Model comparison and evaluation
- Production-ready code practices
- How to communicate technical work

---

## 📊 The Dataset (Week 1, Day 1)

**Use: Twitter Sentiment140 Dataset**
- **Size:** 160,000 tweets (manageable, not overwhelming)
- **Labels:** Binary (positive/negative)
- **Real-world relevance:** Social media sentiment = industry standard
- **Why this over IMDB:** 
  - Shorter text = faster to process
  - More modern than 2011 IMDB reviews
  - Shows you understand messy real data (hashtags, @mentions, typos)

**Download:**
```
Link: https://www.kaggle.com/datasets/kazanova/sentiment140
Or direct download: http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
```

---

## 🗓️ Week-by-Week Breakdown

### **Week 1: Data + Baseline (Days 1-3)**

**Day 1-2: Load & Explore**
- Load 160k tweets (use sample of 50k first for speed)
- Data checks: missing values, class balance, text length distribution
- Clean: remove URLs, mentions, special chars (but keep some for realism)
- **Deliverable:** Jupyter notebook with 3-4 visualizations of your data

**Day 3: Baseline Model**
- Split: 80/20 train/test
- **Model:** Naive Bayes with TF-IDF features
  - Why: Fast to train, interpretable, great baseline
  - Shows you know feature engineering basics
- **Metrics:** Accuracy, Precision, Recall, F1
- **Deliverable:** Confusion matrix + classification report

```python
# Quick baseline template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

baseline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('nb', MultinomialNB())
])

baseline.fit(X_train, y_train)
baseline_score = baseline.score(X_test, y_test)
```

---

### **Week 2: Better Model (Days 4-7)**

**Day 4-5: Logistic Regression + Feature Engineering**
- Swap Naive Bayes for Logistic Regression (better for text, more impressive)
- Add features:
  - TF-IDF (n-grams: unigrams + bigrams)
  - Simple lexical features: avg word length, punctuation count
- Tune hyperparameters (C, max_iter)
- **Deliverable:** Comparison table (Naive Bayes vs Logistic Regression)

**Day 6-7: Simple Neural Network (Optional but Impressive)**
- Use **PyTorch or TensorFlow/Keras** (whichever you're learning)
- Architecture:
  - Embedding layer (pre-trained word vectors optional, but cool)
  - 1-2 hidden layers (64-128 neurons)
  - Output: sigmoid for binary classification
- **Why this impresses:** Shows you can code beyond scikit-learn
- **Deliverable:** Training curves + test performance

---

### **Week 3: Polish & Deploy (Days 8-14)**

**Day 8-9: Error Analysis**
- Find tweets your model gets WRONG
- Analyze patterns: sarcasm? mixed sentiment? poor preprocessing?
- Write 2-3 insights in markdown
- **Why this matters:** Recruiters love error analysis—shows real thinking

**Day 10-11: Documentation**
- Write a **clear README.md** with:
  - Problem statement (1 paragraph)
  - Dataset description
  - How to run the code
  - Results summary + visualizations
  - Limitations & future work
- **Code quality:**
  - Clear variable names
  - Comments on non-obvious sections
  - Separate data, train, eval scripts

**Day 12-13: Simple Web Demo (Optional, +Huge Points)**
- Use **Streamlit** (5 lines of setup):
  ```python
  import streamlit as st
  st.title("Tweet Sentiment Classifier")
  user_input = st.text_input("Enter a tweet:")
  prediction = model.predict([user_input])
  st.write(f"Sentiment: {prediction}")
  ```
- Push to GitHub with deployment instructions
- **Bonus:** Deploy free on Hugging Face Spaces (30 min)

**Day 14: Final Polish**
- Review code for typos/clarity
- Update README with results table
- Commit to GitHub with clear commit messages

---

## 📂 Your GitHub Structure (Recruiters Look at This)

```
sentiment-analysis/
├── README.md                    # Clear, professional
├── requirements.txt             # pip install -r requirements.txt
├── data/
│   ├── raw/                     # Original dataset (gitignored if large)
│   └── processed/               # Cleaned data samples
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_baseline_model.ipynb
│   └── 03_neural_network.ipynb
├── src/
│   ├── preprocess.py            # Text cleaning functions
│   ├── train.py                 # Training pipeline
│   ├── evaluate.py              # Metrics + visualizations
│   └── models.py                # Model definitions
├── results/
│   ├── confusion_matrix.png
│   ├── model_comparison.csv
│   └── training_curves.png
├── app.py                       # Streamlit app (optional)
└── .gitignore
```

---

## 🎯 What Impresses Recruiters (In Order)

1. **Working code on GitHub** ✅ (Must-have)
   - Cloneable, runnable
   - No hard-coded paths
   - Clear requirements.txt

2. **Thoughtful evaluation** ✅ (Shows maturity)
   - Not just accuracy—precision, recall, F1
   - Confusion matrix + error analysis
   - Why your choices matter

3. **Multiple models** ✅ (Shows curiosity)
   - Naive Bayes → Logistic Regression → Neural Net
   - Comparison table with pros/cons

4. **Professional documentation** ✅ (Shows communication)
   - Clear README
   - Markdown, not rambling text
   - How to reproduce your work

5. **One extra feature** ✅ (The differentiator)
   - Web demo (Streamlit)
   - OR error analysis notebook
   - OR deployed model

---

## 💡 Interview-Ready Talking Points

When recruiters ask about your project, say:

> "I built a sentiment classifier on 160k real tweets. Started with Naive Bayes (~81% accuracy) as a baseline, then improved to Logistic Regression (85%) by adding bigrams. Tried a neural network but found it overfitted—the simpler model was actually better. I analyzed failure cases and found sarcasm was the biggest challenge. The whole thing is production-ready on GitHub and deployed on Hugging Face Spaces."

This shows: baseline thinking → iteration → real evaluation → deployment.

---

## 🔴 What NOT to Do (Common Mistakes)

- ❌ Just report accuracy. Show **precision/recall/F1** (for interview credibility)
- ❌ Use 100% of data for training. Always 80/20 split (shows you know validation)
- ❌ Skip preprocessing. Show you cleaned URLs/handles (real-world thinking)
- ❌ Leave code messy. 5 minutes of cleanup = +10% interview impression
- ❌ Stop at Jupyter. Push to GitHub with .py files (shows maturity)

---

## 📚 Quick Learning Resources (If You Get Stuck)

| Topic | Resource | Time |
|-------|----------|------|
| TF-IDF explained | [StatQuest (YouTube)](https://www.youtube.com/watch?v=D2V3FC511QY) | 15 min |
| Logistic Regression | [3Blue1Brown](https://www.youtube.com/watch?v=4qJaSmvT1QE) | 20 min |
| PyTorch basics | [Official tutorials](https://pytorch.org/tutorials/) | 1 hour |
| Streamlit setup | [Official docs](https://docs.streamlit.io/) | 30 min |

---

## 🎁 Bonus Ideas (Add if Time Permits)

- **Real-time Twitter API:** Fetch live tweets and classify them (shows system thinking)
- **Visualization dashboard:** Plot sentiment over time for specific hashtags
- **Comparison with SOTA:** Show how your model compares to a BERT baseline
- **Multi-class sentiment:** Extend to 5-star ratings instead of binary

---

## ✅ Submission Checklist (Before You Share with Recruiters)

- [ ] GitHub repo is public
- [ ] README is clear (no jargon assumed)
- [ ] Code runs: `python train.py` works end-to-end
- [ ] requirements.txt is accurate
- [ ] Results table shows multiple models
- [ ] At least 2 visualizations (confusion matrix, model comparison)
- [ ] Error analysis notebook exists
- [ ] (Optional) Streamlit app deployed
- [ ] No hard-coded paths, no API keys in code
- [ ] Recent commit message (within 2 weeks)

---

## 🚀 After Submission: Internship Pitch

In your cover letter/email:

> "I've built and deployed a tweet sentiment classifier that compares multiple approaches (Naive Bayes, Logistic Regression, Neural Network). The project demonstrates my ability to handle real-world messy data, evaluate models rigorously, and ship working code. It's live on [GitHub link] and deployed on Hugging Face Spaces."

This is **way more impressive** than just "I did a sentiment analysis project."

---

Good luck! You've got this. 🔥
