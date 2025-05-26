#!/bin/bash

# LangChain Application Setup Script

echo "🚀 Setting up LangChain Application..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

echo "✅ Python 3 found"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p chroma_db

# Check for .env file
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating template..."
    cat > .env << EOF
OPENAI_API_KEY=your_openai_api_key_here
SERPAPI_API_KEY=your_serpapi_key_here
# Add your API keys above
EOF
    echo "📝 Please edit .env file with your API keys"
fi

# Run basic validation
echo "🔍 Running validation..."
python3 -c "
import sys
sys.path.append('src')
try:
    from src.utils import validate_environment
    validation = validate_environment()
    print('Environment validation completed')
    for key, status in validation.items():
        symbol = '✅' if status['valid'] else ('⚠️' if not status['required'] else '❌')
        print(f'{symbol} {key}: {\"Valid\" if status[\"valid\"] else \"Not configured\"}')
except Exception as e:
    print(f'Validation error: {e}')
"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. 📝 Edit .env file with your OpenAI API key"
echo "2. 🌐 Run web app: streamlit run app.py"
echo "3. 💻 Or try CLI: python cli.py --mode cli --docs data/ai_overview.txt"
echo "4. 🧪 Run demo: python demo.py"
echo ""
echo "📚 Read README.md for detailed usage instructions"
