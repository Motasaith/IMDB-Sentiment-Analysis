# 🌐 Deploy Your IMDb Sentiment Analyzer Live

Here are several options to make your web app accessible to friends online:

## 🚀 Option 1: Streamlit Community Cloud (Recommended - FREE)

### Steps:
1. **Upload to GitHub:**
   ```bash
   # Create a new repository on GitHub
   # Upload all your project files
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Your app will be live at:**
   `https://[your-username]-[repo-name]-[branch]-[hash].streamlit.app`

### Advantages:
- ✅ Completely FREE
- ✅ Easy to use
- ✅ Automatic updates from GitHub
- ✅ Built specifically for Streamlit

## 🔥 Option 2: Render (FREE with limitations)

### Steps:
1. Create account at [render.com](https://render.com)
2. Connect your GitHub repository
3. Create a new "Web Service"
4. Use these settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

### Advantages:
- ✅ FREE tier available
- ✅ Good performance
- ✅ Custom domains

## 💙 Option 3: Railway (Easy deployment)

### Steps:
1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. Click "Deploy from GitHub repo"
4. Select your repository
5. Railway auto-detects Python and deploys

## ⚡ Option 4: Ngrok (Quick testing - Temporary)

For immediate sharing (temporary link):

```bash
# Install ngrok
pip install pyngrok

# Run your app locally
python -m streamlit run app.py

# In another terminal, expose it
ngrok http 8501
```

This gives you a public URL like: `https://abc123.ngrok.io`

## 📋 Pre-Deployment Checklist

### 1. Update requirements.txt for deployment:
```txt
streamlit>=1.25.0
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0
scikit-learn>=1.1.0
nltk>=3.7
beautifulsoup4>=4.10.0
wordcloud>=1.8.0
joblib>=1.1.0
```

### 2. Create .streamlit/config.toml for production:
```toml
[server]
port = 8501
address = "0.0.0.0"
maxUploadSize = 200

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

### 3. Add a Procfile (for some platforms):
```
web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

## 🎯 Quick Deploy Script

I'll create a script to help with GitHub deployment:
