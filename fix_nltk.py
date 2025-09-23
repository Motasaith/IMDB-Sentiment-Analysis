#!/usr/bin/env python3
"""
NLTK Data Fix Script
This script removes corrupted NLTK data and reinstalls fresh data.
"""

import os
import shutil
import sys
import time

def fix_nltk_data():
    """Fix corrupted NLTK data by removing and reinstalling"""
    print("🔧 Fixing NLTK data corruption issue...")

    # NLTK data path
    nltk_data_path = os.path.expanduser("~") + "/AppData/Roaming/nltk_data"

    print(f"📁 NLTK data path: {nltk_data_path}")

    # Step 1: Remove corrupted data
    if os.path.exists(nltk_data_path):
        print("🗑️  Removing corrupted NLTK data...")
        try:
            shutil.rmtree(nltk_data_path)
            print("✅ Corrupted data removed successfully")
        except Exception as e:
            print(f"❌ Error removing data: {e}")
            return False
    else:
        print("ℹ️  NLTK data directory not found")

    # Step 2: Create fresh directory
    print("📁 Creating fresh NLTK data directory...")
    try:
        os.makedirs(nltk_data_path, exist_ok=True)
        print("✅ Fresh directory created")
    except Exception as e:
        print(f"❌ Error creating directory: {e}")
        return False

    # Step 3: Download fresh NLTK data
    print("⬇️  Downloading fresh NLTK data...")
    try:
        import nltk

        # Download required packages
        packages = ['punkt', 'stopwords', 'wordnet']
        for package in packages:
            print(f"Downloading {package}...")
            nltk.download(package, quiet=True)
            print(f"✅ {package} downloaded")

        print("✅ All NLTK data downloaded successfully!")
        return True

    except Exception as e:
        print(f"❌ Error downloading NLTK data: {e}")
        return False

def test_app_import():
    """Test if the app can be imported successfully"""
    print("\n🧪 Testing app import...")
    try:
        # Add current directory to path
        sys.path.insert(0, '.')

        # Try to import the app module
        import app
        print("✅ App module loaded successfully!")
        return True

    except Exception as e:
        print(f"❌ Error importing app: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting NLTK data fix process...\n")

    # Fix NLTK data
    success = fix_nltk_data()

    if success:
        print("\n" + "="*50)
        print("✅ NLTK data fix completed successfully!")
        print("="*50)

        # Test app import
        test_app_import()

    else:
        print("\n" + "="*50)
        print("❌ NLTK data fix failed!")
        print("Please try running the script again or manually fix the NLTK data.")
        print("="*50)
        sys.exit(1)
