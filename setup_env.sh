#!/usr/bin/env bash
set -e

echo "🚀 Setting up evo_neuro_graph environment..."

if ! command -v python3 &>/dev/null; then
  echo "❌ Python3 not found. Please install Python 3.10+."
  exit 1
fi

echo "📦 Creating virtual environment..."
python3 -m venv env

echo "✅ Virtual environment created."
echo ""
echo "👉 Next steps:"
echo "   1. Run: source env/bin/activate"
echo "   2. Then: pip install -r requirements.txt"
echo ""
exit 0
