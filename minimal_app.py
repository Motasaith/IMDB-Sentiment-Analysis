"""
Minimal IMDb Sentiment Analysis Web App
Works without NLTK dependencies for immediate deployment
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from bs4 import BeautifulSoup
import joblib

# Configure page
st.set_page_config(
    page_title="IMDb Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SimpleSentimentAnalyzer:
    """Simple sentiment analyzer without NLTK dependencies"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        # Basic stopwords without NLTK
        self.stop_words = set([
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
            'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
            'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
            'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
            'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
            'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 'above', 
            'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
            'further', 'then', 'once'
        ])
        
    def load_model(self, model_path="models/best_model.pkl", vectorizer_path="models/best_vectorizer.pkl"):
        """Load trained model and vectorizer"""
        try:
            if os.path.exists(model_path) and os.path.exists(vectorizer_path):
                self.model = joblib.load(model_path)
                self.vectorizer = joblib.load(vectorizer_path)
                return True
            else:
                return False
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    
    def clean_text(self, text):
        """Clean and preprocess text without NLTK"""
        # Remove HTML tags
        text = BeautifulSoup(text, 'html.parser').get_text()
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Simple tokenization
        tokens = text.split()
        
        # Remove stopwords and short words
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(tokens)
    
    def predict_sentiment(self, text):
        """Predict sentiment for a single text"""
        if not self.model or not self.vectorizer:
            return None
            
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Vectorize
        text_vec = self.vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = self.model.predict(text_vec)[0]
        probabilities = self.model.predict_proba(text_vec)[0]
        
        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = max(probabilities)
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'probabilities': {
                'negative': probabilities[0],
                'positive': probabilities[1]
            },
            'cleaned_text': cleaned_text
        }

def train_quick_model():
    """Train a quick lightweight model for immediate use"""
    try:
        with st.spinner("⚡ Training quick model (1-2 minutes)..."):
            from simple_model import create_simple_models
            ok = create_simple_models()
            if ok:
                st.success("✅ Quick model trained and saved!")
                st.rerun()
            else:
                st.error("❌ Quick model training failed.")
                st.info("The dataset might be missing. This is expected on Streamlit Cloud.")
    except Exception as e:
        st.error(f"❌ Quick training failed: {e}")
        st.info("Training requires the IMDB Dataset.csv file which is not available in this deployment.")

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 style="text-align: center; color: #1f77b4;">🎬 IMDb Sentiment Analysis</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = SimpleSentimentAnalyzer()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["🏠 Home", "🤖 Single Prediction", "ℹ️ About"]
        )
        
        st.markdown("---")
        st.markdown("## 🔧 Model Status")
        model_loaded = st.session_state.analyzer.load_model()
        if model_loaded:
            st.success("✅ Model loaded successfully!")
        else:
            st.warning("⚠️ Models not found.")
            st.info("Train a model to enable predictions:")
            if st.button("⚡ Train Quick Model", help="Train a fast model (1-2 minutes)"):
                train_quick_model()
    
    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "🤖 Single Prediction":
        show_single_prediction()
    elif page == "ℹ️ About":
        show_about_page()

def show_home_page():
    """Home page content"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## Welcome to IMDb Sentiment Analyzer! 🎬")
        st.markdown("""
        This application uses machine learning to analyze the sentiment of movie reviews.
        Built as part of a **Big Data Analytics** project.
        
        ### 🚀 Features:
        
        - 🎯 **Single Review Analysis**: Real-time sentiment prediction
        - 🤖 **Machine Learning**: Logistic Regression with TF-IDF
        - 🎨 **Modern UI**: Clean, responsive design
        - ⚡ **Quick Training**: Train models directly in the app
        
        ### 🛠️ How it Works:
        
        1. **Text Preprocessing**: Cleans HTML, removes stopwords
        2. **Feature Extraction**: TF-IDF vectorization  
        3. **Classification**: Logistic Regression model
        4. **Results**: Sentiment + confidence scores
        """)
    
    with col2:
        st.markdown("### 🎯 Quick Start")
        st.markdown("""
        **Ready to analyze reviews?**
        
        1. 🤖 **Single Prediction**: Test individual reviews
        2. ⚡ **Train Model**: If no model is loaded
        3. 📊 **View Results**: Get sentiment + confidence
        """)
        
        # Sample prediction if model is loaded
        if st.session_state.analyzer.model:
            st.markdown("### 🧪 Quick Test")
            if st.button("Test Sample Review"):
                sample = "This movie was absolutely fantastic! Great acting and plot."
                result = st.session_state.analyzer.predict_sentiment(sample)
                if result:
                    sentiment = result['sentiment']
                    confidence = result['confidence']
                    st.success(f"🎬 **{sentiment}** ({confidence:.1%} confidence)")
                    st.info(f"*Sample: {sample[:50]}...*")

def show_single_prediction():
    """Single prediction page"""
    st.markdown("## 🤖 Single Review Analysis")
    
    if not st.session_state.analyzer.model:
        st.error("❌ Model not loaded! Please train a model first using the sidebar.")
        return
    
    # Input area
    st.markdown("### 📝 Enter Your Movie Review")
    review_text = st.text_area(
        "Write or paste a movie review here:",
        height=150,
        placeholder="Example: This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout..."
    )
    
    # Prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🎯 Analyze Sentiment", type="primary", use_container_width=True)
    
    if predict_button and review_text.strip():
        with st.spinner("Analyzing sentiment..."):
            result = st.session_state.analyzer.predict_sentiment(review_text)
            
            if result:
                # Display results
                st.markdown("---")
                st.markdown("### 📊 Analysis Results")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Sentiment result
                    sentiment = result['sentiment']
                    confidence = result['confidence']
                    
                    if sentiment == "Positive":
                        st.markdown(f'<div style="background-color: #d4edda; color: #155724; padding: 1rem; border-radius: 5px; text-align: center; font-size: 2rem; font-weight: bold;">😊 {sentiment}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div style="background-color: #f8d7da; color: #721c24; padding: 1rem; border-radius: 5px; text-align: center; font-size: 2rem; font-weight: bold;">😞 {sentiment}</div>', unsafe_allow_html=True)
                    
                    st.metric("Confidence", f"{confidence:.1%}")
                
                with col2:
                    # Probability breakdown
                    st.markdown("**Probability Breakdown:**")
                    st.write(f"Positive: {result['probabilities']['positive']:.3f}")
                    st.write(f"Negative: {result['probabilities']['negative']:.3f}")
                    
                    # Additional metrics
                    st.markdown("**Text Statistics:**")
                    st.write(f"Original Length: {len(review_text)} chars")
                    st.write(f"Processed Length: {len(result['cleaned_text'])} chars")
                    st.write(f"Word Count: {len(result['cleaned_text'].split())} words")
                
                # Show processed text
                with st.expander("🔍 View Processed Text"):
                    st.text_area("Processed Review:", result['cleaned_text'], height=100, disabled=True)
    
    elif predict_button:
        st.warning("⚠️ Please enter a review to analyze!")
    
    # Sample reviews for testing
    st.markdown("---")
    st.markdown("### 🧪 Try These Sample Reviews")
    
    samples = [
        "This movie was absolutely fantastic! The acting was superb and the cinematography was breathtaking.",
        "What a terrible waste of time. The plot made no sense and the acting was wooden.",
        "An average film. It had its moments but overall nothing special.",
    ]
    
    for i, sample in enumerate(samples):
        if st.button(f"📄 Sample {i+1}", key=f"sample_{i}"):
            st.text_area("Selected review:", sample, height=80, disabled=True, key=f"sample_text_{i}")

def show_about_page():
    """About page content"""
    st.markdown("## ℹ️ About This Application")
    
    st.markdown("""
    ### 🎬 IMDb Sentiment Analysis Project
    
    This is a **Big Data Analytics** project that demonstrates end-to-end machine learning pipeline
    for sentiment analysis of movie reviews.
    
    ### 🛠️ Technical Stack:
    - **Frontend**: Streamlit
    - **ML Library**: Scikit-learn
    - **Text Processing**: BeautifulSoup, Regex
    - **Vectorization**: TF-IDF
    - **Model**: Logistic Regression
    - **Deployment**: Streamlit Cloud
    
    ### 📊 Dataset:
    - **Source**: IMDb Movie Reviews
    - **Size**: 50,000 reviews
    - **Classes**: Positive/Negative sentiment
    - **Format**: CSV with review text and labels
    
    ### 🎯 Model Performance:
    - **Expected Accuracy**: ~88-90%
    - **Training Time**: 1-2 minutes (quick model)
    - **Features**: 5,000 TF-IDF features
    - **Preprocessing**: HTML removal, lowercasing, stopword removal
    
    ### 🚀 Deployment Features:
    - **Cloud-Ready**: No local dependencies required
    - **Auto-Training**: Models can be trained directly in the app
    - **Responsive Design**: Works on desktop and mobile
    - **Real-Time Predictions**: Instant sentiment analysis
    """)
    
    st.markdown("---")
    st.markdown("**Built with ❤️ for Big Data Analytics**")

if __name__ == "__main__":
    main()
