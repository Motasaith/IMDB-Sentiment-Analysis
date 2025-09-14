"""
Complete IMDb Sentiment Analysis Pipeline
Run this script to execute the entire project pipeline
"""

import sys
import os
import subprocess
import time
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print(f"\\n{'='*60}")
    print(f"🚀 {text}")
    print('='*60)

def print_step(step, text):
    """Print formatted step"""
    print(f"\\n{step}. {text}")
    print('-' * 40)

def run_command(command, description=""):
    """Run a command and handle errors"""
    print(f"▶️ Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if description:
            print(f"✅ {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def check_file_exists(filepath, description=""):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"✅ Found: {filepath}")
        return True
    else:
        print(f"❌ Missing: {filepath}")
        if description:
            print(f"   {description}")
        return False

def main():
    """Main pipeline execution"""
    print_header("IMDb SENTIMENT ANALYSIS PIPELINE")
    print("This script will run the complete project pipeline:")
    print("1. Setup and install dependencies")
    print("2. Data preprocessing and EDA")
    print("3. Model training and evaluation")
    print("4. Launch web application")
    
    # Check prerequisites
    print_step(1, "Checking Prerequisites")
    
    if not check_file_exists("IMDB Dataset.csv", "Please ensure the dataset file is in the project directory"):
        print("\\n❌ Cannot continue without the dataset!")
        print("Please download the IMDB Dataset.csv file and place it in this directory.")
        return
    
    # Setup phase
    print_step(2, "Setting Up Environment")
    print("Installing required packages...")
    
    if not run_command("pip install -r requirements.txt", "Dependencies installed"):
        print("⚠️ Some packages might have failed to install. Continuing...")
    
    # Download NLTK data
    print("\nDownloading NLTK data...")
    nltk_script = '''import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True) 
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)
print("NLTK data downloaded successfully!")
'''
    
    with open('temp_nltk.py', 'w', encoding='utf-8') as f:
        f.write(nltk_script)
    
    run_command("python temp_nltk.py", "NLTK data downloaded")
    
    # Clean up
    if os.path.exists('temp_nltk.py'):
        os.remove('temp_nltk.py')
    
    # Data preprocessing phase
    print_step(3, "Data Preprocessing and EDA")
    print("Running data preprocessing pipeline...")
    print("This will:")
    print("- Load and analyze the IMDb dataset")
    print("- Clean and preprocess text data")
    print("- Generate visualizations and word clouds")
    print("- Prepare data for modeling")
    
    if run_command("python data_preprocessing.py", "Data preprocessing completed"):
        print("📊 Generated visualization files:")
        if os.path.exists('eda_analysis.png'):
            print("  - eda_analysis.png")
        if os.path.exists('wordclouds.png'):
            print("  - wordclouds.png")
    else:
        print("⚠️ Data preprocessing had issues, but continuing...")
    
    # Model training phase
    print_step(4, "Machine Learning Model Training")
    print("Training multiple sentiment analysis models...")
    print("This will:")
    print("- Train Logistic Regression, Naive Bayes, SVM, Random Forest")
    print("- Use both TF-IDF and Count Vectorizers")
    print("- Evaluate model performance")
    print("- Save the best performing model")
    
    if run_command("python ml_models.py", "Model training completed"):
        print("🤖 Model training successful!")
        if os.path.exists('models'):
            print("📁 Model files saved in 'models/' directory:")
            for file in os.listdir('models'):
                print(f"  - {file}")
        if os.path.exists('model_evaluation.png'):
            print("📊 Model evaluation plot: model_evaluation.png")
    else:
        print("❌ Model training failed! Check the error messages above.")
        print("You can still run the web app, but predictions won't work.")
    
    # Web application phase
    print_step(5, "Launching Web Application")
    print("Starting Streamlit web application...")
    print("The web app includes:")
    print("- Interactive sentiment analysis")
    print("- Dataset exploration")
    print("- Batch processing")
    print("- Model performance metrics")
    
    print("\\n🌐 Opening web application...")
    print("If it doesn't open automatically, go to: http://localhost:8501")
    print("\\nPress Ctrl+C to stop the application")
    
    # Launch Streamlit
    try:
        subprocess.run("streamlit run app.py", shell=True, check=True)
    except KeyboardInterrupt:
        print("\\n\\n🛑 Application stopped by user")
    except Exception as e:
        print(f"\\n❌ Error launching Streamlit: {e}")
        print("\\nYou can manually start the app with: streamlit run app.py")
    
    print_header("PIPELINE COMPLETED")
    print("✅ Project setup complete!")
    print("\\n📁 Generated files:")
    files_to_check = [
        'eda_analysis.png',
        'wordclouds.png', 
        'model_evaluation.png',
        'models/best_model.pkl',
        'models/best_vectorizer.pkl',
        'models/model_results.csv'
    ]
    
    for file in files_to_check:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} (missing)")
    
    print("\\n🚀 To restart the web app anytime, run: streamlit run app.py")

if __name__ == "__main__":
    main()
