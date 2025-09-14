"""
IMDb Dataset Preprocessing and Exploratory Data Analysis
Phase 1: Data Science Pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)
    
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK punkt_tab tokenizer...")
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading NLTK wordnet...")
    nltk.download('wordnet', quiet=True)

# Set style for plots
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        pass  # Use default style
        
try:
    sns.set_palette("husl")
except:
    pass

class IMDbDataProcessor:
    """Class to handle IMDb dataset preprocessing and analysis"""
    
    def __init__(self, csv_path="IMDB Dataset.csv"):
        """Initialize with dataset path"""
        self.csv_path = csv_path
        self.df = None
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def load_data(self):
        """Load the IMDb dataset"""
        print("📊 Loading IMDb dataset...")
        self.df = pd.read_csv(self.csv_path)
        print(f"✅ Dataset loaded successfully!")
        print(f"📈 Dataset shape: {self.df.shape}")
        print(f"📋 Columns: {list(self.df.columns)}")
        return self.df
    
    def basic_info(self):
        """Display basic information about the dataset"""
        if self.df is None:
            print("❌ Please load data first!")
            return
            
        print("\\n" + "="*50)
        print("📊 DATASET OVERVIEW")
        print("="*50)
        print(f"Shape: {self.df.shape}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\\n📋 Data Types:")
        print(self.df.dtypes)
        
        print("\\n🔍 Missing Values:")
        missing = self.df.isnull().sum()
        print(missing[missing > 0] if missing.sum() > 0 else "No missing values!")
        
        print("\\n📈 Sentiment Distribution:")
        sentiment_counts = self.df['sentiment'].value_counts()
        print(sentiment_counts)
        print(f"Balance ratio: {sentiment_counts.min()/sentiment_counts.max():.3f}")
        
        print("\\n📝 Sample Reviews:")
        for i, (idx, row) in enumerate(self.df.sample(2).iterrows()):
            print(f"\\n{i+1}. [{row['sentiment']}] {row['review'][:200]}...")
    
    def clean_text(self, text):
        """Clean individual text review"""
        try:
            # Remove HTML tags
            text = BeautifulSoup(text, 'html.parser').get_text()
            
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters and digits (keep spaces)
            text = re.sub(r'[^a-zA-Z\s]', ' ', text)
            
            # Remove extra whitespaces
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Tokenize
            try:
                tokens = word_tokenize(text)
            except:
                # Fallback to simple splitting if word_tokenize fails
                tokens = text.split()
            
            # Remove stopwords and lemmatize
            try:
                tokens = [self.lemmatizer.lemmatize(word) for word in tokens 
                         if word not in self.stop_words and len(word) > 2]
            except:
                # Fallback without lemmatization
                tokens = [word for word in tokens 
                         if word not in self.stop_words and len(word) > 2]
            
            return ' '.join(tokens)
            
        except Exception as e:
            print(f"Error cleaning text: {e}")
            # Return basic cleaned text
            return re.sub(r'[^a-zA-Z\s]', ' ', str(text).lower())
    
    def preprocess_data(self):
        """Clean and preprocess all reviews"""
        if self.df is None:
            print("❌ Please load data first!")
            return
            
        print("🧹 Cleaning and preprocessing reviews...")
        
        # Clean reviews
        self.df['cleaned_review'] = self.df['review'].apply(self.clean_text)
        
        # Add review length features
        self.df['review_length'] = self.df['review'].apply(len)
        self.df['cleaned_length'] = self.df['cleaned_review'].apply(len)
        self.df['word_count'] = self.df['cleaned_review'].apply(lambda x: len(x.split()))
        
        # Convert sentiment to binary
        self.df['sentiment_binary'] = (self.df['sentiment'] == 'positive').astype(int)
        
        print("✅ Preprocessing completed!")
        print(f"📊 Average original length: {self.df['review_length'].mean():.0f}")
        print(f"📊 Average cleaned length: {self.df['cleaned_length'].mean():.0f}")
        print(f"📊 Average word count: {self.df['word_count'].mean():.0f}")
        
        return self.df
    
    def create_visualizations(self, save_plots=True):
        """Create comprehensive EDA visualizations"""
        if self.df is None or 'cleaned_review' not in self.df.columns:
            print("❌ Please preprocess data first!")
            return
            
        print("📊 Creating visualizations...")
        
        # Set up the plotting area
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Sentiment Distribution
        plt.subplot(2, 3, 1)
        sentiment_counts = self.df['sentiment'].value_counts()
        colors = ['#ff6b6b', '#4ecdc4']
        plt.pie(sentiment_counts.values, labels=sentiment_counts.index, 
               autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('Sentiment Distribution', fontsize=14, fontweight='bold')
        
        # 2. Review Length Distribution
        plt.subplot(2, 3, 2)
        plt.hist(self.df['review_length'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(self.df['review_length'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {self.df["review_length"].mean():.0f}')
        plt.xlabel('Review Length (characters)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Review Lengths')
        plt.legend()
        
        # 3. Word Count by Sentiment
        plt.subplot(2, 3, 3)
        sns.boxplot(data=self.df, x='sentiment', y='word_count', palette=['#ff6b6b', '#4ecdc4'])
        plt.title('Word Count Distribution by Sentiment')
        plt.ylabel('Word Count')
        
        # 4. Review Length by Sentiment
        plt.subplot(2, 3, 4)
        sns.violinplot(data=self.df, x='sentiment', y='review_length', palette=['#ff6b6b', '#4ecdc4'])
        plt.title('Review Length Distribution by Sentiment')
        plt.ylabel('Character Count')
        
        # 5. Word Count Histogram
        plt.subplot(2, 3, 5)
        plt.hist(self.df[self.df['sentiment']=='positive']['word_count'], 
                bins=30, alpha=0.7, label='Positive', color='#4ecdc4')
        plt.hist(self.df[self.df['sentiment']=='negative']['word_count'], 
                bins=30, alpha=0.7, label='Negative', color='#ff6b6b')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        plt.title('Word Count Distribution by Sentiment')
        plt.legend()
        
        # 6. Correlation Heatmap
        plt.subplot(2, 3, 6)
        corr_data = self.df[['review_length', 'cleaned_length', 'word_count', 'sentiment_binary']].corr()
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create Word Clouds
        self.create_wordclouds(save_plots)
        
    def create_wordclouds(self, save_plots=True):
        """Create word clouds for positive and negative reviews"""
        print("☁️ Creating word clouds...")
        
        try:
            # Separate positive and negative reviews
            positive_reviews = self.df[self.df['sentiment']=='positive']['cleaned_review'].dropna()
            negative_reviews = self.df[self.df['sentiment']=='negative']['cleaned_review'].dropna()
            
            # Take a sample for word cloud generation (for performance)
            positive_text = ' '.join(positive_reviews.head(1000))
            negative_text = ' '.join(negative_reviews.head(1000))
            
            # Check if we have enough text
            if len(positive_text.strip()) < 50 or len(negative_text.strip()) < 50:
                print("⚠️ Not enough text data for word clouds after cleaning.")
                return
            
            # Create word clouds
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Positive wordcloud
            try:
                wordcloud_pos = WordCloud(
                    width=800, height=400,
                    background_color='white',
                    colormap='Greens',
                    max_words=50,
                    min_font_size=10
                ).generate(positive_text)
                
                ax1.imshow(wordcloud_pos, interpolation='bilinear')
                ax1.set_title('Positive Reviews Word Cloud', fontsize=16, fontweight='bold', color='green')
                ax1.axis('off')
            except Exception as e:
                print(f"⚠️ Could not create positive wordcloud: {e}")
                ax1.text(0.5, 0.5, 'Positive\nWordCloud\nUnavailable', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax1.transAxes, fontsize=16, color='green')
                ax1.axis('off')
            
            # Negative wordcloud
            try:
                wordcloud_neg = WordCloud(
                    width=800, height=400,
                    background_color='white',
                    colormap='Reds',
                    max_words=50,
                    min_font_size=10
                ).generate(negative_text)
                
                ax2.imshow(wordcloud_neg, interpolation='bilinear')
                ax2.set_title('Negative Reviews Word Cloud', fontsize=16, fontweight='bold', color='red')
                ax2.axis('off')
            except Exception as e:
                print(f"⚠️ Could not create negative wordcloud: {e}")
                ax2.text(0.5, 0.5, 'Negative\nWordCloud\nUnavailable', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax2.transAxes, fontsize=16, color='red')
                ax2.axis('off')
            
            plt.tight_layout()
            if save_plots:
                plt.savefig('wordclouds.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"❌ Error creating word clouds: {e}")
            print("Skipping word cloud generation...")
        
    def get_top_words(self, n=20):
        """Get top words for each sentiment"""
        if 'cleaned_review' not in self.df.columns:
            print("❌ Please preprocess data first!")
            return
            
        print(f"🔝 Top {n} words by sentiment:")
        
        for sentiment in ['positive', 'negative']:
            text = ' '.join(self.df[self.df['sentiment']==sentiment]['cleaned_review'])
            words = text.split()
            top_words = Counter(words).most_common(n)
            
            print(f"\\n{sentiment.upper()} REVIEWS:")
            print("-" * 30)
            for word, count in top_words:
                print(f"{word:15} : {count:,}")
                
        return top_words
    
    def prepare_for_modeling(self, test_size=0.2, random_state=42):
        """Prepare data for machine learning models"""
        if 'cleaned_review' not in self.df.columns:
            print("❌ Please preprocess data first!")
            return None, None, None, None
            
        print("🎯 Preparing data for modeling...")
        
        X = self.df['cleaned_review']
        y = self.df['sentiment_binary']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"✅ Data split completed!")
        print(f"📊 Training set: {len(X_train):,} samples")
        print(f"📊 Test set: {len(X_test):,} samples")
        print(f"📊 Positive ratio in train: {y_train.mean():.3f}")
        print(f"📊 Positive ratio in test: {y_test.mean():.3f}")
        
        return X_train, X_test, y_train, y_test

def main():
    """Main function to run the preprocessing pipeline"""
    print("🚀 Starting IMDb Data Preprocessing Pipeline...")
    
    # Initialize processor
    processor = IMDbDataProcessor()
    
    # Load and explore data
    processor.load_data()
    processor.basic_info()
    
    # Preprocess data
    processor.preprocess_data()
    
    # Create visualizations
    processor.create_visualizations()
    
    # Get top words
    processor.get_top_words()
    
    # Prepare for modeling
    X_train, X_test, y_train, y_test = processor.prepare_for_modeling()
    
    print("\\n🎉 Data preprocessing completed successfully!")
    print("📁 Generated files:")
    print("  - eda_analysis.png")
    print("  - wordclouds.png")
    
    return processor, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    processor, X_train, X_test, y_train, y_test = main()
