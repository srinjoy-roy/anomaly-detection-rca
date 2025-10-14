#!/usr/bin/env python3
"""
Setup script for Spark Cluster Anomaly Detection System
Capstone Project - IIT Roorkee Applied Data Science & AI
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

def setup_environment():
    """Setup the project environment"""
    print("🔥 Setting up Spark Cluster Anomaly Detection System")
    print("=" * 60)
    
    # Create necessary directories
    directories = [
        "output",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Check if virtual environment exists
    venv_path = Path(".venv")
    if not venv_path.exists():
        print("\n🔧 Creating virtual environment...")
        try:
            subprocess.check_call([sys.executable, "-m", "venv", ".venv"])
            print("✅ Virtual environment created")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error creating virtual environment: {e}")
            return False
    
    # Install core dependencies in virtual environment
    print("\n📦 Installing core dependencies in virtual environment...")
    try:
        # Use the virtual environment's pip
        venv_pip = ".venv/bin/pip" if os.name != 'nt' else ".venv\\Scripts\\pip.exe"
        core_packages = [
            "pandas", "numpy", "scikit-learn", "matplotlib", 
            "plotly", "streamlit", "networkx", "pyyaml", "tqdm", "joblib"
        ]
        subprocess.check_call([venv_pip, "install"] + core_packages)
        print("✅ Core dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        print("💡 You can manually install with: source .venv/bin/activate && pip install pandas numpy scikit-learn matplotlib plotly streamlit networkx pyyaml tqdm joblib")
        return False
    
    # Download NLTK data (if needed)
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK data downloaded")
    except Exception as e:
        print(f"⚠️ Warning: Could not download NLTK data: {e}")
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the main pipeline: python src/main_pipeline.py")
    print("2. Launch dashboard: streamlit run src/dashboard/streamlit_app.py")
    
    return True

def run_quick_test():
    """Run a quick test of the system"""
    print("\n🧪 Running quick system test...")
    
    try:
        # Test imports
        sys.path.append('src')
        from data.log_parser import SparkLogParser
        from models.anomaly_detector import SparkLogAnomalyDetector
        from models.root_cause_analyzer import RootCauseAnalyzer
        
        print("✅ All modules imported successfully")
        
        # Test log parser initialization
        parser = SparkLogParser()
        print("✅ Log parser initialized")
        
        # Test anomaly detector initialization
        detector = SparkLogAnomalyDetector()
        print("✅ Anomaly detector initialized")
        
        # Test root cause analyzer initialization
        analyzer = RootCauseAnalyzer()
        print("✅ Root cause analyzer initialized")
        
        print("🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = setup_environment()
    
    if success:
        test_success = run_quick_test()
        if test_success:
            print("\n🚀 System is ready to use!")
        else:
            print("\n⚠️ Setup completed but tests failed. Check dependencies.")
    else:
        print("\n❌ Setup failed. Please check error messages above.")
        sys.exit(1)
