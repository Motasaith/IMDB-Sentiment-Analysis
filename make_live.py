"""
Quick Live Deployment Script
This script makes your Streamlit app instantly accessible to anyone with the link
"""

import subprocess
import sys
import time
import webbrowser
from threading import Thread

def install_ngrok():
    """Install pyngrok if not installed"""
    try:
        import pyngrok
        return True
    except ImportError:
        print("📦 Installing ngrok...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyngrok"])
            return True
        except Exception as e:
            print(f"❌ Failed to install ngrok: {e}")
            return False

def run_streamlit():
    """Run Streamlit app"""
    print("🚀 Starting Streamlit app...")
    try:
        # Run streamlit in the background
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "app.py", 
            "--server.headless", "true"
        ])
        return process
    except Exception as e:
        print(f"❌ Failed to start Streamlit: {e}")
        return None

def create_tunnel():
    """Create ngrok tunnel"""
    try:
        from pyngrok import ngrok
        
        print("🌐 Creating public tunnel...")
        # Create tunnel to port 8501 (Streamlit default)
        tunnel = ngrok.connect(8501)
        public_url = tunnel.public_url
        
        print(f"\n🎉 Your app is now LIVE!")
        print("="*60)
        print(f"🔗 Public URL: {public_url}")
        print("="*60)
        print("📱 Share this link with your friends!")
        print("⏰ This link will work as long as this script is running")
        print("\n🛑 Press Ctrl+C to stop")
        
        return public_url, tunnel
        
    except Exception as e:
        print(f"❌ Failed to create tunnel: {e}")
        return None, None

def main():
    """Main function"""
    print("🌟 IMDb Sentiment Analyzer - Live Deployment")
    print("="*50)
    
    # Install ngrok if needed
    if not install_ngrok():
        print("❌ Cannot proceed without ngrok")
        return
    
    # Start Streamlit
    streamlit_process = run_streamlit()
    if not streamlit_process:
        print("❌ Cannot start Streamlit app")
        return
    
    # Wait for Streamlit to start
    print("⏳ Waiting for Streamlit to start...")
    time.sleep(5)
    
    # Create tunnel
    public_url, tunnel = create_tunnel()
    if not public_url:
        print("❌ Cannot create public tunnel")
        streamlit_process.terminate()
        return
    
    # Try to open the URL in browser
    try:
        webbrowser.open(public_url)
    except:
        pass
    
    try:
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
        
        # Clean up
        try:
            tunnel.close()
            streamlit_process.terminate()
            print("✅ Cleanup completed")
        except:
            pass

if __name__ == "__main__":
    main()
