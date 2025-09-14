"""
Simple Setup Script for IMDb Sentiment Analysis Project
This installs packages one by one to avoid dependency conflicts
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a single package"""
    print(f"Installing {package}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def main():
    """Install packages step by step"""
    print("🚀 Installing packages for IMDb Sentiment Analysis...")
    
    # Core packages
    packages = [
        "pandas",
        "numpy", 
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "nltk",
        "beautifulsoup4",
        "wordcloud",
        "streamlit",
        "plotly",
        "joblib"
    ]
    
    failed_packages = []
    
    for package in packages:
        if not install_package(package):
            failed_packages.append(package)
    
    print(f"\n{'='*50}")
    print("📊 Installation Summary")
    print('='*50)
    
    if failed_packages:
        print(f"❌ Failed packages: {', '.join(failed_packages)}")
        print("\n⚠️ You may need to:")
        print("1. Install Microsoft Visual C++ Build Tools")
        print("2. Try installing failed packages individually")
        print("3. Use conda instead of pip for some packages")
    else:
        print("✅ All packages installed successfully!")
    
    # Download NLTK data
    print("\n📥 Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        print("✅ NLTK data downloaded!")
    except Exception as e:
        print(f"❌ NLTK download failed: {e}")
    
    print("\n🎉 Setup complete! You can now run:")
    print("   python data_preprocessing.py")
    print("   python ml_models.py") 
    print("   streamlit run app.py")

if __name__ == "__main__":
    main()
