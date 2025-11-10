#!/usr/bin/env bash
set -e  # stop on first error

echo "🚀 Setting up evo_neuro_graph environment..."

# 1. Detect Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.10+ first."
    exit 1
fi

# 2. Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv env

# 3. Activate environment
if [[ "$OSTYPE" == "darwin"* ]]; then
    source env/bin/activate
else
    source env/bin/activate
fi

# 4. Upgrade pip
pip install --upgrade pip

# 5. Install dependencies
echo "📥 Installing Python packages..."
pip install -r requirements.txt

# 6. Verify directory structure
if [ ! -d "experiments" ] || [ ! -d "eng" ] || [ ! -d "tasks" ]; then
    echo "⚠️  Warning: expected folders (experiments, eng, tasks) not found."
fi

# 7. Optional: create output/artifacts folders
mkdir -p artifacts logs

# 8. Test installation
echo "🧪 Testing import..."
python - <<'PY'
try:
    import numpy, torch
    print("✅ Environment OK — core packages loaded.")
except Exception as e:
    print("❌ Something failed:", e)
PY

echo "✅ Setup complete!"
echo ""
echo "👉 Next steps:"
echo "source env/bin/activate"
echo "python -m experiments.train_tinygrid --policy mlp --pop_size 64 --episodes 8 --max_steps 128 --generations 20"
