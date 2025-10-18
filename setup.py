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
    print("Setting up Spark Cluster Anomaly Detection System")
    print("=" * 60)

    # Create necessary directories
    directories = [
        "output",
        "logs"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

    # Check if virtual environment exists
    venv_path = Path(".venv")
    if not venv_path.exists():
        print("\nCreating virtual environment...")
        try:
            subprocess.check_call([sys.executable, "-m", "venv", ".venv"])
            print("Virtual environment created")
        except subprocess.CalledProcessError as e:
            print(f"Error creating virtual environment: {e}")
            return False

    # Install dependencies in virtual environment (require requirements.txt)
    print("\nInstalling dependencies in virtual environment...")
    try:
        # Use the virtual environment's pip
        venv_pip = ".venv/bin/pip" if os.name != 'nt' else ".venv\\Scripts\\pip.exe"
        req_file = Path("requirements.txt")
        if not req_file.exists():
            print("requirements.txt not found. Aborting setup.")
            print("Please create a requirements.txt at the project root and rerun: python setup.py")
            return False
        print("Found requirements.txt — installing with -r requirements.txt")
        subprocess.check_call([venv_pip, "install", "-r", str(req_file)])
        print("Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        print("You can manually install with: source .venv/bin/activate && pip install -r requirements.txt")
        return False

    # Download NLTK data (if needed)
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("NLTK data downloaded")
    except Exception as e:
        print(f"Warning: Could not download NLTK data: {e}")

    print("\nSetup completed successfully!")
    print("\nNext steps:")
    print("1. Run the main pipeline: python src/main_pipeline.py")
    print("2. Launch dashboard: streamlit run src/dashboard/streamlit_app.py")

    return True


def get_venv_python() -> str:
    """Return the path to the virtual environment's Python executable."""
    return ".venv/bin/python" if os.name != 'nt' else ".venv\\Scripts\\python.exe"


if __name__ == "__main__":
    success = setup_environment()

    if success:
        print("\nSystem is ready to use!")
    else:
        print("\nSetup failed. Please check error messages above.")
        sys.exit(1)
