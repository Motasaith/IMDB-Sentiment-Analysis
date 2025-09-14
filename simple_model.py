"""
Simple Fallback Sentiment Analysis Model
This provides basic sentiment analysis functionality when full models aren't trained yet
"""

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

class SimpleSentimentModel:
    """Simple and fast sentiment analysis model"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.is_trained = False
        
    def clean_text(self, text):
        """Basic text cleaning"""
        # Convert to lowercase
        text = str(text).lower()
        # Remove special characters
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def train_quick_model(self, sample_size=5000):
        """Train a quick model with a subset of data"""
        print("🚀 Training quick sentiment model...")
        
        # Load dataset
        if not os.path.exists("IMDB Dataset.csv"):
            print("❌ Dataset not found!")
            return False
            
        try:
            df = pd.read_csv("IMDB Dataset.csv")
            
            # Use a smaller sample for quick training
            df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
            
            # Clean text
            df_sample['cleaned_review'] = df_sample['review'].apply(self.clean_text)
            
            # Prepare data
            X = df_sample['cleaned_review']
            y = (df_sample['sentiment'] == 'positive').astype(int)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Create vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # Vectorize
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_test_vec = self.vectorizer.transform(X_test)
            
            # Train model
            self.model = LogisticRegression(random_state=42)
            self.model.fit(X_train_vec, y_train)
            
            # Test accuracy
            y_pred = self.model.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"✅ Quick model trained with {accuracy:.3f} accuracy")
            self.is_trained = True
            
            # Save model
            os.makedirs("models", exist_ok=True)
            joblib.dump(self.model, "models/simple_model.pkl")
            joblib.dump(self.vectorizer, "models/simple_vectorizer.pkl")
            
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
    
    def load_model(self):
        """Load the simple model"""
        try:
            if os.path.exists("models/simple_model.pkl") and os.path.exists("models/simple_vectorizer.pkl"):
                self.model = joblib.load("models/simple_model.pkl")
                self.vectorizer = joblib.load("models/simple_vectorizer.pkl")
                self.is_trained = True
                return True
        except:
            pass
        return False
    
    def predict_sentiment(self, text):
        """Predict sentiment for text"""
        if not self.is_trained:
            if not self.load_model():
                if not self.train_quick_model():
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

def create_simple_models():
    """Create simple models for immediate use"""
    print("Creating simple fallback models...")
    
    simple_model = SimpleSentimentModel()
    if simple_model.train_quick_model():
        # Also save as best_model for compatibility
        try:
            import shutil
            shutil.copy("models/simple_model.pkl", "models/best_model.pkl")
            shutil.copy("models/simple_vectorizer.pkl", "models/best_vectorizer.pkl")
            print("✅ Simple models created successfully!")
            return True
        except:
            pass
    
    return False

if __name__ == "__main__":
    create_simple_models()
