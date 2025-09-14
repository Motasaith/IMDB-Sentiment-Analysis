"""
Setup script for IMDb Sentiment Analysis Project
Run this first to install all dependencies and download required data
"""

import subprocess
import sys
import nltk

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✅ Packages installed successfully!")

def download_nltk_data():
    """Download required NLTK data"""
    print("Downloading NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('vader_lexicon')
    print("✅ NLTK data downloaded successfully!")

def download_spacy_model():
    """Download spaCy English model"""
    print("Downloading spaCy English model...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    print("✅ spaCy model downloaded successfully!")

if __name__ == "__main__":
    print("🚀 Setting up IMDb Sentiment Analysis Project...")
    
    try:
        install_requirements()
        download_nltk_data()
        download_spacy_model()
        print("\n🎉 Setup completed successfully!")
        print("You can now run: streamlit run app.py")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        print("Please install packages manually or check your internet connection.")
