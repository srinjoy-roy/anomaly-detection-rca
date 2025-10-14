#!/bin/bash
# Activation script for Spark Cluster Anomaly Detection System
# Usage: source activate_env.sh

echo "🔥 Activating Spark Cluster Anomaly Detection Environment"
echo "=========================================================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run setup.py first."
    return 1
fi

# Activate virtual environment
source .venv/bin/activate

echo "✅ Virtual environment activated"
echo "Python path: $(which python)"
echo ""
echo "🚀 Available commands:"
echo "  python src/main_pipeline.py          # Run complete pipeline"
echo "  streamlit run src/dashboard/streamlit_app.py  # Launch dashboard"
echo "  python setup.py                      # Re-run setup if needed"
echo ""
echo "📁 Project structure:"
echo "  src/                 # Core system code"
echo "  logs/Spark/          # Your Spark cluster logs"
echo "  output/              # Generated results"
echo "  docs/                # Documentation & presentation"
echo ""
echo "🎯 Ready for project!"
