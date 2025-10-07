#!/bin/bash
set -e
echo "🚀 Setting up pilot experiment environment..."

# Check Python version
python_version=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $required_version+ required, found $python_version"
    exit 1
fi
echo "✅ Python $python_version"

# Create directories
mkdir -p data/raw/lanl data/processed
mkdir -p experiments/{phase0,phase1,phase2,phase3,phase4}
mkdir -p logs models

# Install requirements
echo "📦 Installing dependencies..."
if pip install -q -r requirements.txt; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    echo ""
    echo "This system uses an externally managed Python environment."
    echo "Please install dependencies manually or use a virtual environment:"
    echo ""
    echo "Option 1 - Manual install:"
    echo "  pip install --break-system-packages -r requirements.txt"
    echo ""
    echo "Option 2 - Virtual environment:"
    echo "  python -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    echo ""
    echo "For now, continuing with setup..."
fi

# Download NLTK data
echo "📚 Downloading NLTK data..."
python -c "
try:
    import nltk
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_context

    # Download
    for package in ['punkt', 'punkt_tab']:
        try:
            nltk.download(package, quiet=True)
            print(f'✅ {package}')
        except Exception as e:
            print(f'⚠️ {package}: {e}')
except ImportError:
    print('⚠️ NLTK not installed yet - skipping NLTK data download')
    print('   Install dependencies first, then run: python -c \"import nltk; nltk.download(\\\"punkt\\\")\"')
"

# Check LANL data
echo ""
echo "🔍 Checking for LANL dataset..."
if [ -f "data/raw/lanl/auth.txt.gz" ] || [ -f "data/raw/lanl/auth.txt" ]; then
    echo "✅ Auth file found"
else
    echo "❌ LANL dataset not found!"
    echo ""
    echo "Download from: https://csr.lanl.gov/data/cyber1/"
    echo "Files needed:"
    echo " - auth.txt.gz (or auth.txt)"
    echo " - redteam.txt"
    echo "Place in: data/raw/lanl/"
    exit 1
fi

if [ -f "data/raw/lanl/redteam.txt" ]; then
    echo "✅ Red team labels found"
else
    echo "⚠️ Red team labels missing (optional but recommended)"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo " 1. python scripts/phase0_validate.py"
echo " 2. python scripts/phase1_context.py"
