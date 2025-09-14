"""
IMDb Sentiment Analysis Web App
Phase 2: Interactive Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="IMDb Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .positive {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .negative {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

class SentimentAnalyzer:
    """Class to handle sentiment analysis functionality"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
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
        """Clean and preprocess text"""
        # Remove HTML tags
        text = BeautifulSoup(text, 'html.parser').get_text()
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\\s]', '', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\\s+', ' ', text).strip()
        
        # Tokenize
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        
        # Remove stopwords and lemmatize
        try:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens 
                     if word not in self.stop_words and len(word) > 2]
        except:
            tokens = [word for word in tokens 
                     if word not in self.stop_words and len(word) > 2]
        
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
    
    def predict_batch(self, texts):
        """Predict sentiment for multiple texts"""
        results = []
        for text in texts:
            result = self.predict_sentiment(text)
            if result:
                results.append(result)
        return results

def load_dataset():
    """Load and cache the IMDb dataset"""
    @st.cache_data
    def _load_data():
        if os.path.exists("IMDB Dataset.csv"):
            return pd.read_csv("IMDB Dataset.csv")
        return None
    
    return _load_data()

def create_wordcloud(text, title, colormap='viridis'):
    """Create word cloud visualization"""
    wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        colormap=colormap,
        max_words=100
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')
    
    return fig

def train_quick_model():
    """Train a quick lightweight model for immediate use"""
    try:
        with st.spinner("⚡ Training quick model (1-2 minutes)..."):
            from simple_model import create_simple_models
            ok = create_simple_models()
            if ok:
                st.success("✅ Quick model trained and saved!")
                # Reload model into analyzer
                if 'analyzer' in st.session_state:
                    st.session_state.analyzer.load_model()
            else:
                st.error("❌ Quick model training failed.")
                st.info("You can try Full Training or run training locally and commit the models/ folder.")
    except Exception as e:
        st.error(f"❌ Quick training failed: {e}")


def train_models_automatically():
    """Train models automatically within the web app"""
    try:
        with st.spinner("🔄 Training ML models... This may take 5-10 minutes."):
            # Create progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Load and preprocess data
            status_text.text("📊 Loading and preprocessing data...")
            progress_bar.progress(20)
            
            from data_preprocessing import IMDbDataProcessor
            processor = IMDbDataProcessor()
            processor.load_data()
            processor.preprocess_data()
            X_train, X_test, y_train, y_test = processor.prepare_for_modeling()
            
            # Step 2: Train models
            status_text.text("🤖 Training multiple ML models...")
            progress_bar.progress(40)
            
            from ml_models import SentimentModels
            sentiment_models = SentimentModels()
            sentiment_models.create_models()
            
            # Step 3: Train and evaluate
            status_text.text("📈 Training and evaluating models...")
            progress_bar.progress(70)
            
            results = sentiment_models.train_and_evaluate(X_train, X_test, y_train, y_test)
            
            # Step 4: Save models
            status_text.text("💾 Saving trained models...")
            progress_bar.progress(90)
            
            sentiment_models.save_models()
            
            # Complete
            progress_bar.progress(100)
            status_text.text("✅ Model training completed successfully!")
            
            st.success("🎉 Models trained and saved successfully!")
            st.info("Please refresh the page to use the trained models.")
            
            # Show training results
            st.markdown("### 📊 Training Results")
            best_key, best_result = sentiment_models.find_best_model()
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Best Model", best_key.replace('_', ' ').title())
            with col2:
                st.metric("Accuracy", f"{best_result['accuracy']:.3f}")
            with col3:
                st.metric("F1-Score", f"{best_result['f1']:.3f}")
            with col4:
                st.metric("AUC", f"{best_result['auc']:.3f}")
            
    except Exception as e:
        st.error(f"❌ Training failed: {str(e)}")
        st.info("The training process requires significant computational resources. You may need to run this locally first.")

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🎬 IMDb Sentiment Analysis</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = SentimentAnalyzer()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["🏠 Home", "📊 Dataset Analysis", "🤖 Single Prediction", "📝 Batch Analysis", "📈 Model Performance"]
        )
        
        st.markdown("---")
        st.markdown("## ℹ️ About")
        st.markdown("""
        This app analyzes movie review sentiments using machine learning.
        
        **Features:**
        - Single review analysis
        - Batch processing
        - Interactive visualizations
        - Model performance metrics
        """)
        
        # Model status
        st.markdown("---")
        st.markdown("## 🔧 Model Status")
        model_loaded = st.session_state.analyzer.load_model()
        if model_loaded:
            st.success("✅ Model loaded successfully!")
        else:
            st.warning("⚠️ Full models not found.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("⚡ Quick Model", help="Train a fast model (1-2 minutes)"):
                    train_quick_model()
            with col2:
                if st.button("🚀 Full Training", help="Train all models (5-10 minutes)"):
                    train_models_automatically()
    
    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Dataset Analysis":
        show_dataset_analysis()
    elif page == "🤖 Single Prediction":
        show_single_prediction()
    elif page == "📝 Batch Analysis":
        show_batch_analysis()
    elif page == "📈 Model Performance":
        show_model_performance()

def show_home_page():
    """Home page content"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## Welcome to IMDb Sentiment Analyzer! 🎬")
        st.markdown("""
        This application uses advanced machine learning techniques to analyze the sentiment of movie reviews.
        Built as part of a comprehensive **Big Data Analytics** project, it demonstrates the complete pipeline
        from data preprocessing to model deployment.
        
        ### 🚀 Project Features:
        
        **Phase 1: Data Science**
        - 📊 **Dataset Analysis**: Explore 50,000 IMDb movie reviews
        - 🧹 **Data Preprocessing**: Clean text, remove HTML, handle stopwords
        - 📈 **Exploratory Data Analysis**: Word clouds, distributions, correlations
        - 🤖 **Machine Learning**: Multiple models (Logistic Regression, Naive Bayes, SVM, Random Forest)
        
        **Phase 2: Web Application**  
        - 🎯 **Single Review Analysis**: Real-time sentiment prediction
        - 📝 **Batch Processing**: Analyze multiple reviews at once
        - 📊 **Interactive Visualizations**: Dynamic charts and word clouds
        - 🎨 **Modern UI**: Clean, responsive design with Streamlit
        
        **Phase 3: Model Performance**
        - 📈 **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, AUC
        - 🏆 **Model Comparison**: Side-by-side performance analysis
        - 📋 **Detailed Reports**: Classification reports and confusion matrices
        """)
        
        # Quick stats
        dataset = load_dataset()
        if dataset is not None:
            st.markdown("### 📊 Dataset Overview")
            col1_1, col1_2, col1_3, col1_4 = st.columns(4)
            
            with col1_1:
                st.metric("Total Reviews", f"{len(dataset):,}")
            with col1_2:
                positive_count = len(dataset[dataset['sentiment'] == 'positive'])
                st.metric("Positive Reviews", f"{positive_count:,}")
            with col1_3:
                negative_count = len(dataset[dataset['sentiment'] == 'negative'])
                st.metric("Negative Reviews", f"{negative_count:,}")
            with col1_4:
                avg_length = dataset['review'].str.len().mean()
                st.metric("Avg Review Length", f"{avg_length:.0f} chars")
    
    with col2:
        st.markdown("### 🎯 Quick Start")
        st.markdown("""
        **Ready to analyze some reviews?**
        
        1. 🤖 **Single Prediction**: Test individual reviews
        2. 📝 **Batch Analysis**: Upload CSV files
        3. 📊 **Dataset Analysis**: Explore the data
        4. 📈 **Model Performance**: View metrics
        """)
        
        # Sample predictions
        if st.session_state.analyzer.model:
            st.markdown("### 🧪 Sample Predictions")
            
            sample_reviews = [
                "This movie was absolutely fantastic!",
                "Terrible plot and bad acting.",
                "An okay film, nothing special."
            ]
            
            for i, review in enumerate(sample_reviews):
                result = st.session_state.analyzer.predict_sentiment(review)
                if result:
                    sentiment = result['sentiment']
                    confidence = result['confidence']
                    
                    if sentiment == "Positive":
                        st.markdown(f'<div class="positive">😊 {sentiment} ({confidence:.1%})</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="negative">😞 {sentiment} ({confidence:.1%})</div>', unsafe_allow_html=True)
                    
                    st.markdown(f"*\"{review}\"*")
                    st.markdown("---")

def show_dataset_analysis():
    """Dataset analysis page"""
    st.markdown("## 📊 Dataset Analysis")
    
    dataset = load_dataset()
    if dataset is None:
        st.error("❌ Dataset not found! Please ensure 'IMDB Dataset.csv' is in the project folder.")
        return
    
    # Add a button to process data if needed
    if st.button("🔄 Process Data for Visualization", help="This will clean and prepare data for analysis"):
        with st.spinner("Processing dataset... This may take a few minutes."):
            try:
                # Import and run preprocessing
                from data_preprocessing import IMDbDataProcessor
                processor = IMDbDataProcessor()
                processor.load_data()
                processor.preprocess_data()
                
                # Store processed data in session state
                st.session_state.processed_data = processor.df
                st.success("✅ Data processed successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error processing data: {e}")
                return
    
    # Basic statistics
    st.markdown("### 📈 Basic Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reviews", f"{len(dataset):,}")
    with col2:
        positive_pct = (dataset['sentiment'] == 'positive').mean() * 100
        st.metric("Positive %", f"{positive_pct:.1f}%")
    with col3:
        avg_length = dataset['review'].str.len().mean()
        st.metric("Avg Length", f"{avg_length:.0f}")
    with col4:
        unique_words = len(set(' '.join(dataset['review'].head(1000)).split()))
        st.metric("Unique Words", f"{unique_words:,}")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🥧 Sentiment Distribution")
        sentiment_counts = dataset['sentiment'].value_counts()
        
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Distribution of Sentiments",
            color_discrete_map={'positive': '#2E8B57', 'negative': '#DC143C'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📏 Review Length Distribution")
        dataset['review_length'] = dataset['review'].str.len()
        
        fig = px.histogram(
            dataset, 
            x='review_length',
            nbins=50,
            title="Distribution of Review Lengths",
            labels={'review_length': 'Review Length (characters)', 'count': 'Frequency'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Word clouds
    st.markdown("### ☁️ Word Clouds")
    col1, col2 = st.columns(2)
    
    analyzer = SentimentAnalyzer()
    
    with col1:
        st.markdown("#### Positive Reviews")
        positive_text = ' '.join(dataset[dataset['sentiment']=='positive']['review'].head(500))
        cleaned_positive = analyzer.clean_text(positive_text)
        
        if cleaned_positive:
            fig = create_wordcloud(cleaned_positive, "Positive Reviews", 'Greens')
            st.pyplot(fig)
    
    with col2:
        st.markdown("#### Negative Reviews")
        negative_text = ' '.join(dataset[dataset['sentiment']=='negative']['review'].head(500))
        cleaned_negative = analyzer.clean_text(negative_text)
        
        if cleaned_negative:
            fig = create_wordcloud(cleaned_negative, "Negative Reviews", 'Reds')
            st.pyplot(fig)
    
    # Sample reviews
    st.markdown("### 📝 Sample Reviews")
    sample_df = dataset.sample(5)
    
    for _, row in sample_df.iterrows():
        sentiment = row['sentiment']
        review = row['review']
        
        if sentiment == 'positive':
            st.markdown(f'<div class="positive">😊 Positive</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="negative">😞 Negative</div>', unsafe_allow_html=True)
        
        st.markdown(f"*{review[:300]}{'...' if len(review) > 300 else ''}*")
        st.markdown("---")

def show_single_prediction():
    """Single prediction page"""
    st.markdown("## 🤖 Single Review Analysis")
    
    if not st.session_state.analyzer.model:
        st.error("❌ Model not loaded! Please train the model first.")
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
                        st.markdown(f'<div class="positive" style="text-align: center; font-size: 2rem;">😊 {sentiment}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="negative" style="text-align: center; font-size: 2rem;">😞 {sentiment}</div>', unsafe_allow_html=True)
                    
                    st.metric("Confidence", f"{confidence:.1%}")
                
                with col2:
                    # Probability chart
                    fig = go.Figure(data=[
                        go.Bar(
                            x=['Negative', 'Positive'],
                            y=[result['probabilities']['negative'], result['probabilities']['positive']],
                            marker_color=['#DC143C', '#2E8B57']
                        )
                    ])
                    fig.update_layout(
                        title="Prediction Probabilities",
                        yaxis_title="Probability",
                        showlegend=False,
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed breakdown
                st.markdown("### 🔍 Detailed Breakdown")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Original Review:**")
                    st.text_area("", review_text, height=100, disabled=True)
                
                with col2:
                    st.markdown("**Processed Review:**")
                    st.text_area("", result['cleaned_text'], height=100, disabled=True)
                
                # Additional metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Length", len(review_text))
                with col2:
                    st.metric("Processed Length", len(result['cleaned_text']))
                with col3:
                    st.metric("Word Count", len(result['cleaned_text'].split()))
    
    elif predict_button:
        st.warning("⚠️ Please enter a review to analyze!")
    
    # Sample reviews for testing
    st.markdown("---")
    st.markdown("### 🧪 Try These Sample Reviews")
    
    samples = [
        "This movie was absolutely fantastic! The acting was superb, the plot was engaging, and the cinematography was breathtaking. I would definitely recommend it to anyone.",
        "What a terrible waste of time. The plot made no sense, the acting was wooden, and I fell asleep halfway through. Avoid at all costs.",
        "An average film. It had its moments but overall nothing special. Some good scenes but the pacing was off.",
        "Mind-blowing experience! This film changed my perspective on cinema. Every scene was crafted perfectly and the emotional depth was incredible.",
        "Boring and predictable. I've seen this story a hundred times before. The director clearly ran out of ideas."
    ]
    
    for i, sample in enumerate(samples):
        if st.button(f"📄 Sample {i+1}", key=f"sample_{i}"):
            st.text_area("Selected review:", sample, height=80, disabled=True, key=f"sample_text_{i}")

def show_batch_analysis():
    """Batch analysis page"""
    st.markdown("## 📝 Batch Analysis")
    
    if not st.session_state.analyzer.model:
        st.error("❌ Model not loaded! Please train the model first.")
        return
    
    st.markdown("### 📤 Upload Reviews")
    
    # File upload options
    tab1, tab2 = st.tabs(["📄 CSV Upload", "✏️ Text Input"])
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="CSV should have a 'review' column"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                if 'review' not in df.columns:
                    st.error("❌ CSV must contain a 'review' column!")
                else:
                    st.success(f"✅ Loaded {len(df)} reviews!")
                    
                    # Show preview
                    st.markdown("### 👀 Data Preview")
                    st.dataframe(df.head())
                    
                    # Analyze button
                    if st.button("🚀 Analyze All Reviews", type="primary"):
                        analyze_batch(df['review'].tolist())
                        
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
    
    with tab2:
        st.markdown("Enter multiple reviews (one per line):")
        text_input = st.text_area(
            "",
            height=200,
            placeholder="This movie was great!\\nTerrible plot and acting.\\nAn okay film overall."
        )
        
        if st.button("🚀 Analyze Reviews", type="primary") and text_input.strip():
            reviews = [line.strip() for line in text_input.split('\\n') if line.strip()]
            analyze_batch(reviews)

def analyze_batch(reviews):
    """Analyze batch of reviews"""
    if not reviews:
        st.warning("⚠️ No reviews to analyze!")
        return
    
    with st.spinner(f"Analyzing {len(reviews)} reviews..."):
        results = st.session_state.analyzer.predict_batch(reviews)
    
    if not results:
        st.error("❌ Analysis failed!")
        return
    
    # Create results dataframe
    df_results = pd.DataFrame({
        'Review': reviews[:len(results)],
        'Sentiment': [r['sentiment'] for r in results],
        'Confidence': [r['confidence'] for r in results],
        'Positive_Prob': [r['probabilities']['positive'] for r in results],
        'Negative_Prob': [r['probabilities']['negative'] for r in results]
    })
    
    # Display summary
    st.markdown("---")
    st.markdown("### 📊 Batch Analysis Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    positive_count = len(df_results[df_results['Sentiment'] == 'Positive'])
    negative_count = len(df_results[df_results['Sentiment'] == 'Negative'])
    avg_confidence = df_results['Confidence'].mean()
    
    with col1:
        st.metric("Total Reviews", len(df_results))
    with col2:
        st.metric("Positive", positive_count)
    with col3:
        st.metric("Negative", negative_count)
    with col4:
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment distribution
        fig = px.pie(
            df_results,
            names='Sentiment',
            title="Sentiment Distribution",
            color_discrete_map={'Positive': '#2E8B57', 'Negative': '#DC143C'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Confidence distribution
        fig = px.histogram(
            df_results,
            x='Confidence',
            color='Sentiment',
            title="Confidence Distribution",
            color_discrete_map={'Positive': '#2E8B57', 'Negative': '#DC143C'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Results table
    st.markdown("### 📋 Detailed Results")
    st.dataframe(df_results, use_container_width=True)
    
    # Download results
    csv = df_results.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="sentiment_analysis_results.csv",
        mime="text/csv"
    )

def show_model_performance():
    """Model performance page"""
    st.markdown("## 📈 Model Performance")
    
    # Check if model results exist
    if os.path.exists("models/model_results.csv"):
        results_df = pd.read_csv("models/model_results.csv", index_col=0)
        
        st.markdown("### 🏆 Model Comparison")
        
        # Display metrics table
        st.dataframe(results_df.round(4), use_container_width=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy comparison
            fig = px.bar(
                results_df.reset_index(),
                x='index',
                y='accuracy',
                title="Model Accuracy Comparison",
                labels={'index': 'Model', 'accuracy': 'Accuracy'}
            )
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # F1 score comparison
            fig = px.bar(
                results_df.reset_index(),
                x='index',
                y='f1',
                title="Model F1 Score Comparison",
                labels={'index': 'Model', 'f1': 'F1 Score'}
            )
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Best model highlight
        best_model = results_df.loc[results_df['f1'].idxmax()]
        st.markdown("### 🥇 Best Performing Model")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Model", results_df['f1'].idxmax())
        with col2:
            st.metric("Accuracy", f"{best_model['accuracy']:.3f}")
        with col3:
            st.metric("Precision", f"{best_model['precision']:.3f}")
        with col4:
            st.metric("Recall", f"{best_model['recall']:.3f}")
        with col5:
            st.metric("F1-Score", f"{best_model['f1']:.3f}")
        
        # Metrics heatmap
        st.markdown("### 🔥 Performance Heatmap")
        fig = px.imshow(
            results_df.values,
            labels=dict(x="Metrics", y="Models", color="Score"),
            x=results_df.columns,
            y=results_df.index,
            aspect="auto",
            color_continuous_scale="RdYlBu_r"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("⚠️ Model performance data not found!")
        st.info("Please train the models first by running: `python ml_models.py`")
        
        # Show example performance data
        st.markdown("### 📊 Expected Performance Metrics")
        example_data = {
            'Model': ['Logistic Regression + TF-IDF', 'Naive Bayes + TF-IDF', 'SVM + TF-IDF', 'Random Forest + TF-IDF'],
            'Expected Accuracy': ['~89%', '~86%', '~88%', '~87%'],
            'Expected F1-Score': ['~89%', '~86%', '~88%', '~87%'],
            'Training Time': ['Fast', 'Very Fast', 'Slow', 'Medium']
        }
        st.table(pd.DataFrame(example_data))

if __name__ == "__main__":
    try:
        # Download NLTK data if not present
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        st.info("📥 Downloading required language data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
    
    main()
