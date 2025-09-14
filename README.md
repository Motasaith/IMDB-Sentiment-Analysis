# 🎬 IMDb Sentiment Analysis - Complete BDA Project

A comprehensive **Big Data Analytics** project that implements sentiment analysis on movie reviews using machine learning and deploys it as an interactive web application.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [Results](#results)
- [Deployment](#deployment)
- [Contributing](#contributing)

## 🚀 Project Overview

This project demonstrates a complete machine learning pipeline from raw data to deployed web application:

### **Phase 1: Data Science**
- **Dataset**: 50,000 IMDb movie reviews
- **Preprocessing**: Text cleaning, HTML removal, stopword filtering
- **EDA**: Word clouds, sentiment distributions, statistical analysis
- **Modeling**: Multiple ML algorithms with comprehensive evaluation

### **Phase 2: Web Application**
- **Interactive UI**: Modern Streamlit interface
- **Real-time Predictions**: Instant sentiment analysis
- **Batch Processing**: Analyze multiple reviews
- **Visualizations**: Dynamic charts and insights

### **Phase 3: Production Ready**
- **Model Persistence**: Save/load trained models
- **Performance Metrics**: Comprehensive evaluation
- **Deployment Ready**: Easy setup and configuration

## ✨ Features

### 🔍 **Data Analysis**
- Comprehensive exploratory data analysis
- Interactive visualizations and word clouds
- Statistical insights and patterns
- Data quality assessment

### 🤖 **Machine Learning**
- **Models**: Logistic Regression, Naive Bayes, SVM, Random Forest
- **Vectorization**: TF-IDF and Count Vectorizers
- **Evaluation**: Accuracy, Precision, Recall, F1-Score, AUC
- **Model Comparison**: Side-by-side performance analysis

### 🌐 **Web Application**
- **Single Review Analysis**: Real-time sentiment prediction
- **Batch Processing**: Upload CSV files or paste multiple reviews
- **Interactive Dashboard**: Dataset exploration and insights
- **Model Performance**: Detailed metrics and comparisons
- **Responsive Design**: Works on desktop and mobile

### 📊 **Visualizations**
- Sentiment distribution pie charts
- Review length histograms
- Word clouds (positive/negative)
- Model performance comparisons
- Confidence score distributions
- ROC curves and confusion matrices

## 📁 Project Structure

```
BDA Project/
├── 📊 IMDB Dataset.csv           # IMDb movie reviews dataset
├── 📋 requirements.txt           # Python dependencies
├── 🛠️ setup.py                  # Setup and installation script
├── 🏃 run_pipeline.py           # Complete pipeline execution
├── 📖 README.md                 # This documentation file
│
├── 🔬 Data Science & ML/
│   ├── data_preprocessing.py     # Data cleaning and EDA
│   └── ml_models.py              # Model training and evaluation
│
├── 🌐 Web Application/
│   └── app.py                   # Streamlit web interface
│
├── 📁 Generated Files/
│   ├── models/                  # Saved ML models and results
│   │   ├── best_model.pkl
│   │   ├── best_vectorizer.pkl
│   │   └── model_results.csv
│   ├── eda_analysis.png         # EDA visualizations
│   ├── wordclouds.png           # Word cloud plots
│   └── model_evaluation.png     # Model performance plots
└── 📝 Documentation/
    └── (additional docs if needed)
```

## ⚡ Quick Start

### 1️⃣ **One-Click Setup** (Recommended)
```bash
# Run the complete pipeline
python run_pipeline.py
```
This script will:
- Install all dependencies
- Process the dataset
- Train all models
- Launch the web application

### 2️⃣ **Manual Setup**
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run data preprocessing
python data_preprocessing.py

# 3. Train models
python ml_models.py

# 4. Launch web app
streamlit run app.py
```

## 💻 Installation

### Prerequisites
- Python 3.8 or higher
- 4GB+ RAM (for model training)
- IMDb Dataset.csv file in project directory

### Step-by-Step Installation

1. **Clone/Download the project**
   ```bash
   # Ensure you have the project files in 'BDA Project' directory
   cd "BDA Project"
   ```

2. **Verify dataset**
   ```bash
   # Make sure IMDB Dataset.csv is present
   ls -la "IMDB Dataset.csv"
   ```

3. **Install Python packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data** (automatic on first run)
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords') 
   nltk.download('wordnet')
   ```

## 🎯 Usage

### Web Application Features

#### 🏠 **Home Page**
- Project overview and quick statistics
- Sample predictions
- Quick navigation guide

#### 📊 **Dataset Analysis**
- Sentiment distribution visualization
- Review length analysis
- Word clouds for positive/negative reviews
- Sample review exploration

#### 🤖 **Single Prediction**
- Text input for movie reviews
- Real-time sentiment analysis
- Confidence scores and probabilities
- Text preprocessing visualization
- Pre-loaded sample reviews

#### 📝 **Batch Analysis**
- CSV file upload support
- Multi-line text input
- Batch processing results
- Downloadable analysis reports
- Statistical summaries

#### 📈 **Model Performance**
- Model comparison metrics
- Performance visualizations
- Best model identification
- Detailed evaluation reports

### Command Line Usage

#### Data Preprocessing
```bash
python data_preprocessing.py
```
- Loads and analyzes the dataset
- Generates EDA visualizations
- Prepares data for modeling

#### Model Training
```bash
python ml_models.py
```
- Trains multiple ML models
- Evaluates performance
- Saves best performing model
- Generates evaluation plots

#### Web Application
```bash
streamlit run app.py
```
- Launches the web interface
- Accessible at http://localhost:8501

## 🔧 Technical Details

### Data Preprocessing
- **Text Cleaning**: HTML tag removal, special character handling
- **Tokenization**: NLTK word tokenization
- **Stopword Removal**: English stopwords filtering
- **Lemmatization**: Word normalization
- **Feature Engineering**: Review length, word count metrics

### Machine Learning Models

| Model | Vectorizer | Expected Accuracy | Training Speed |
|-------|------------|------------------|----------------|
| Logistic Regression | TF-IDF | ~89% | Fast |
| Naive Bayes | TF-IDF | ~86% | Very Fast |
| SVM | TF-IDF | ~88% | Slow |
| Random Forest | TF-IDF | ~87% | Medium |

### Performance Metrics
- **Accuracy**: Overall correctness
- **Precision**: True positive rate
- **Recall**: Sensitivity 
- **F1-Score**: Harmonic mean of precision and recall
- **AUC**: Area under ROC curve

### Technology Stack
- **Backend**: Python, scikit-learn, NLTK, pandas, numpy
- **Frontend**: Streamlit, Plotly, matplotlib, seaborn
- **ML**: TF-IDF vectorization, multiple classification algorithms
- **Visualization**: Word clouds, interactive charts, heatmaps

## 📊 Results

### Expected Performance
- **Best Model**: Logistic Regression + TF-IDF
- **Accuracy**: ~89%
- **F1-Score**: ~89%
- **Processing Speed**: ~1000 reviews/second
- **Model Size**: ~50MB

### Generated Outputs
1. **EDA Analysis**: Comprehensive dataset visualization
2. **Word Clouds**: Visual representation of frequent words
3. **Model Evaluation**: Performance comparison charts
4. **Trained Models**: Saved for production use
5. **Interactive Dashboards**: Web-based exploration tools

## 🌐 Deployment

### Local Deployment
```bash
streamlit run app.py
```

### Cloud Deployment Options

#### **Streamlit Cloud** (Recommended)
1. Push code to GitHub repository
2. Connect to Streamlit Cloud
3. Deploy directly from repository

#### **Render**
1. Create `render.yaml` configuration
2. Deploy using Render platform
3. Automatic scaling and HTTPS

#### **Heroku**
1. Create `Procfile` and `runtime.txt`
2. Deploy using Heroku CLI
3. Scale as needed

### Environment Variables
```bash
# For production deployment
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

## 🛠️ Development

### Project Architecture
```
Data Layer → Processing Layer → ML Layer → Application Layer
    ↓              ↓              ↓              ↓
CSV File → Preprocessing → Models → Streamlit App
```

### Adding New Models
1. Implement in `ml_models.py`
2. Add to model dictionary
3. Update evaluation pipeline
4. Test with existing framework

### Customizing the Web App
1. Modify `app.py` for UI changes
2. Add new pages in the main function
3. Update navigation in sidebar
4. Test responsiveness

## 📈 Future Enhancements

### Planned Features
- [ ] Deep Learning models (BERT, LSTM)
- [ ] Multi-language support
- [ ] Real-time streaming analysis
- [ ] Advanced visualization options
- [ ] Model explainability features
- [ ] A/B testing framework
- [ ] API endpoints for integration
- [ ] Mobile app version

### Performance Optimizations
- [ ] Model quantization
- [ ] Caching strategies
- [ ] Batch processing optimization
- [ ] GPU acceleration support

## 🤝 Contributing

### How to Contribute
1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Test thoroughly
5. Submit a pull request

### Areas for Contribution
- Model improvements
- UI/UX enhancements
- Documentation updates
- Bug fixes and optimizations
- New feature development

## 📄 License

This project is created for educational purposes as part of a Big Data Analytics course. Feel free to use and modify for learning and non-commercial purposes.

## 👨‍💻 Authors

**Your Name**
- Course: Big Data Analytics
- Project: IMDb Sentiment Analysis
- Contact: [your.email@example.com]

## 🙏 Acknowledgments

- IMDb for providing the dataset
- Scikit-learn team for ML algorithms
- Streamlit team for the web framework
- NLTK contributors for text processing tools
- Open source community for various libraries

## 📞 Support

If you encounter any issues:

1. **Check Prerequisites**: Ensure Python 3.8+ and all dependencies are installed
2. **Verify Dataset**: Make sure `IMDB Dataset.csv` is in the project directory
3. **Run Pipeline**: Try the automated `python run_pipeline.py`
4. **Check Logs**: Look for error messages in the console
5. **Manual Steps**: Run each script individually to isolate issues

### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| "Dataset not found" | Ensure `IMDB Dataset.csv` is in project root |
| "Module not found" | Run `pip install -r requirements.txt` |
| "NLTK data missing" | Run the setup script or download manually |
| "Model not loaded" | Train models first with `python ml_models.py` |
| "Streamlit won't start" | Check if port 8501 is available |

---

**🎉 Enjoy exploring sentiment analysis with this comprehensive BDA project!**

For the latest updates and detailed documentation, refer to the project files and comments within the code.
