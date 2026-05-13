"""
Sentiment Analysis Web App - Streamlit
Week 3: Deployment (OPTIONAL but impressive)

Installation:
    pip install streamlit

Run locally:
    streamlit run app.py

Deploy to Hugging Face Spaces (free, 5 minutes):
    1. Create account: https://huggingface.co
    2. Create new Space (Streamlit template)
    3. Push this file + requirements.txt to the repo
    4. Done! Your app is live.

Why this impresses recruiters:
    - Shows you can deploy ML models
    - Professional-looking interface
    - Real people can test your model
"""

import streamlit as st
import torch
import pickle
import numpy as np
import re
from datetime import datetime

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Tweet Sentiment Classifier",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .positive {
        color: #28a745;
        font-weight: bold;
    }
    .negative {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL AND VECTORIZER
# ============================================================================

@st.cache_resource
def load_model():
    """Load pre-trained model and vectorizer"""
    try:
        # Load neural network model
        model = torch.load('neural_network_model.pt', map_location='cpu')
        
        # Load vectorizer
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model files not found! Please train the model first.")
        return None, None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clean_text(text):
    """Clean tweet text"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s\.\!\?]', '', text)
    text = ' '.join(text.split())
    return text


def predict_sentiment(text, model, vectorizer, device='cpu'):
    """Predict sentiment of input text"""
    # Clean and vectorize
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned]).toarray()
    
    # Convert to tensor
    features_tensor = torch.FloatTensor(features)
    
    # Predict
    model.eval()
    with torch.no_grad():
        pred = model(features_tensor).numpy()[0][0]
    
    sentiment = 'Positive' if pred > 0.5 else 'Negative'
    confidence = float(pred) if pred > 0.5 else float(1 - pred)
    
    return sentiment, confidence, cleaned


# ============================================================================
# MAIN APP
# ============================================================================

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🎯 Tweet Sentiment Classifier")
    st.markdown("*Classifies tweets as positive or negative using machine learning*")

with col2:
    st.markdown(f"**Model Version:** 1.0  \n**Last Updated:** 2024")

st.divider()

# Load model
model_state, vectorizer = load_model()

if model_state is None or vectorizer is None:
    st.error("⚠️ Unable to load model. Please ensure model files are available.")
    st.stop()

# Initialize model properly
from sentiment_neural_network import SentimentNeuralNet
input_dim = vectorizer.get_feature_names_out().shape[0]
model = SentimentNeuralNet(input_dim=input_dim)
model.load_state_dict(model_state)

# ============================================================================
# SIDEBAR: INFORMATION
# ============================================================================

with st.sidebar:
    st.markdown("### 📊 Model Information")
    st.info("""
    **Architecture:** Neural Network
    - Input: TF-IDF features (5000)
    - Hidden Layer 1: 256 neurons
    - Hidden Layer 2: 128 neurons
    - Output: Binary classification
    
    **Training Data:** 50,000 tweets
    **Accuracy:** ~82%
    """)
    
    st.divider()
    
    st.markdown("### 📚 How It Works")
    st.markdown("""
    1. **Input:** Paste a tweet or text
    2. **Processing:** Text is cleaned and converted to numerical features
    3. **Prediction:** Neural network predicts sentiment
    4. **Output:** Positive/Negative with confidence score
    """)
    
    st.divider()
    
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Works best with tweets and social media text
    - More accurate with 10+ words
    - Handles hashtags and mentions
    - Struggles with heavy sarcasm
    """)

# ============================================================================
# MAIN INPUT SECTION
# ============================================================================

st.markdown("### ✍️ Enter Text to Classify")

# Two input methods
tab1, tab2 = st.tabs(["Single Tweet", "Batch Upload"])

with tab1:
    # Single text input
    user_input = st.text_area(
        "Paste a tweet or text:",
        placeholder="e.g., 'I absolutely love this product! Highly recommend!' 🚀",
        height=100
    )
    
    col_submit, col_clear = st.columns(2)
    
    with col_submit:
        if st.button("🔍 Analyze Sentiment", key="single"):
            if user_input.strip():
                sentiment, confidence, cleaned = predict_sentiment(
                    user_input, model, vectorizer
                )
                
                # Display results
                st.divider()
                st.markdown("### 📈 Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if sentiment == 'Positive':
                        st.markdown(f"<p class='positive'>✅ {sentiment}</p>", 
                                   unsafe_allow_html=True)
                    else:
                        st.markdown(f"<p class='negative'>❌ {sentiment}</p>", 
                                   unsafe_allow_html=True)
                
                with col2:
                    st.metric("Confidence", f"{confidence:.2%}")
                
                with col3:
                    if sentiment == 'Positive':
                        sentiment_emoji = "😊 Positive"
                    else:
                        sentiment_emoji = "😞 Negative"
                    st.markdown(f"**Prediction:** {sentiment_emoji}")
                
                # Show processed text
                with st.expander("ℹ️ View Processed Text"):
                    st.markdown(f"**Original:** {user_input}")
                    st.markdown(f"**Cleaned:** {cleaned}")
                
                # Confidence visualization
                st.markdown("### Confidence Breakdown")
                confidence_dict = {
                    'Positive': confidence if sentiment == 'Positive' else 1 - confidence,
                    'Negative': 1 - confidence if sentiment == 'Positive' else confidence
                }
                st.bar_chart(confidence_dict)
            else:
                st.warning("Please enter some text to analyze.")

with tab2:
    st.markdown("""
    **Batch Analysis:** Upload a CSV with a 'text' column to classify multiple tweets at once.
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            import pandas as pd
            df = pd.read_csv(uploaded_file)
            
            if 'text' not in df.columns:
                st.error("CSV must contain a 'text' column")
            else:
                if st.button("🔍 Analyze All Tweets", key="batch"):
                    st.markdown("### Processing...")
                    progress_bar = st.progress(0)
                    
                    results = []
                    for idx, text in enumerate(df['text']):
                        sentiment, confidence, _ = predict_sentiment(
                            str(text), model, vectorizer
                        )
                        results.append({
                            'text': text,
                            'sentiment': sentiment,
                            'confidence': confidence
                        })
                        progress_bar.progress((idx + 1) / len(df['text']))
                    
                    results_df = pd.DataFrame(results)
                    
                    st.markdown("### 📊 Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name=f"sentiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        positive_count = (results_df['sentiment'] == 'Positive').sum()
                        st.metric("Positive Tweets", positive_count)
                    with col2:
                        negative_count = (results_df['sentiment'] == 'Negative').sum()
                        st.metric("Negative Tweets", negative_count)
                    with col3:
                        avg_confidence = results_df['confidence'].mean()
                        st.metric("Avg Confidence", f"{avg_confidence:.2%}")
        
        except Exception as e:
            st.error(f"Error processing file: {e}")

# ============================================================================
# FOOTER & EXAMPLES
# ============================================================================

st.divider()

st.markdown("### 💬 Example Tweets")

example_col1, example_col2, example_col3 = st.columns(3)

examples = [
    ("I love this! Best decision ever! 🎉", "Positive"),
    ("Terrible experience, would not recommend.", "Negative"),
    ("It's okay, nothing special but decent.", "Mixed")
]

for i, (text, expected) in enumerate(examples):
    with [example_col1, example_col2, example_col3][i]:
        st.markdown(f"**Example {i+1}:**")
        st.text(text)
        if st.button(f"Test Example {i+1}", key=f"example_{i}"):
            sentiment, confidence, _ = predict_sentiment(text, model, vectorizer)
            st.success(f"**Prediction:** {sentiment} ({confidence:.2%})")
            if expected != "Mixed":
                match = "✅" if sentiment == expected else "❌"
                st.caption(f"{match} Expected: {expected}")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()

footer_cols = st.columns([3, 1])
with footer_cols[0]:
    st.markdown("""
    ---
    **Sentiment Analysis Classifier**  
    Built with PyTorch | Deployed with Streamlit  
    *A portfolio project by [Your Name]*
    """)

with footer_cols[1]:
    st.markdown("""
    [GitHub](https://github.com/yourname/sentiment-analysis) | 
    [LinkedIn](https://linkedin.com/in/yourname)
    """)

# Optional: Session state to remember inputs
if 'history' not in st.session_state:
    st.session_state.history = []
