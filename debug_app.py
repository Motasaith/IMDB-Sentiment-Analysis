"""
Debug version of IMDb Sentiment Analysis App
Minimal version to test deployment issues
"""

import streamlit as st
import pandas as pd
import os

# Configure page
st.set_page_config(
    page_title="IMDb Sentiment Analyzer - Debug",
    page_icon="🎬",
    layout="wide"
)

def main():
    st.title("🎬 IMDb Sentiment Analysis - Debug Mode")
    
    st.info("🔧 This is a debug version to test deployment...")
    
    # Test basic imports
    try:
        import numpy as np
        st.success("✅ NumPy imported successfully")
    except Exception as e:
        st.error(f"❌ NumPy error: {e}")
    
    try:
        import matplotlib.pyplot as plt
        st.success("✅ Matplotlib imported successfully")
    except Exception as e:
        st.error(f"❌ Matplotlib error: {e}")
    
    try:
        import seaborn as sns
        st.success("✅ Seaborn imported successfully") 
    except Exception as e:
        st.error(f"❌ Seaborn error: {e}")
    
    try:
        import plotly.express as px
        st.success("✅ Plotly imported successfully")
    except Exception as e:
        st.error(f"❌ Plotly error: {e}")
    
    try:
        import sklearn
        st.success("✅ Scikit-learn imported successfully")
    except Exception as e:
        st.error(f"❌ Scikit-learn error: {e}")
    
    # Test NLTK
    st.markdown("### NLTK Status")
    try:
        import nltk
        st.success("✅ NLTK imported successfully")
        
        # Try downloading data
        try:
            nltk.data.find('tokenizers/punkt')
            st.success("✅ NLTK punkt data available")
        except:
            st.info("📥 Downloading punkt...")
            nltk.download('punkt', quiet=True)
            st.success("✅ NLTK punkt downloaded")
            
        try:
            nltk.data.find('corpora/stopwords')
            st.success("✅ NLTK stopwords data available")
        except:
            st.info("📥 Downloading stopwords...")
            nltk.download('stopwords', quiet=True)
            st.success("✅ NLTK stopwords downloaded")
            
        try:
            nltk.data.find('corpora/wordnet')
            st.success("✅ NLTK wordnet data available")
        except:
            st.info("📥 Downloading wordnet...")
            nltk.download('wordnet', quiet=True)
            st.success("✅ NLTK wordnet downloaded")
            
    except Exception as e:
        st.error(f"❌ NLTK error: {e}")
    
    # Test dataset
    st.markdown("### Dataset Status")
    if os.path.exists("IMDB Dataset.csv"):
        st.success("✅ IMDB Dataset.csv found")
        try:
            df = pd.read_csv("IMDB Dataset.csv")
            st.success(f"✅ Dataset loaded: {len(df)} rows")
            st.dataframe(df.head(3))
        except Exception as e:
            st.error(f"❌ Error loading dataset: {e}")
    else:
        st.warning("⚠️ IMDB Dataset.csv not found")
        st.info("This is expected on Streamlit Cloud. The full app will show training options.")
    
    # Test simple functionality
    st.markdown("### Simple Test")
    test_text = st.text_input("Enter some text to test:", "This is a test")
    if test_text:
        st.write(f"You entered: {test_text}")
        st.write(f"Length: {len(test_text)}")
        st.write(f"Words: {len(test_text.split())}")
    
    st.markdown("---")
    st.success("🎉 Debug test completed! If you see this, basic functionality works.")

if __name__ == "__main__":
    main()
