#!/bin/bash

# Setup script for the Real-time Bidirectional Translation Service

set -e

echo "========================================="
echo "Real-time Translation Service Setup"
echo "========================================="

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.11+ is required (found $python_version)"
    exit 1
fi
echo "✓ Python $python_version"

# Check for required commands
echo "Checking dependencies..."
commands=("docker" "docker-compose" "pip")
for cmd in "${commands[@]}"; do
    if ! command -v $cmd &> /dev/null; then
        echo "Error: $cmd is not installed"
        exit 1
    fi
    echo "✓ $cmd installed"
done

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "✓ .env file created"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env file with your API keys before running the service"
else
    echo "✓ .env file exists"
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p data logs ssl alembic/versions
echo "✓ Directories created"

# Install Python dependencies (optional for local development)
read -p "Install Python dependencies locally? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
    echo "✓ Dependencies installed"
fi

# Build Docker images
read -p "Build Docker images? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Building Docker images..."
    docker-compose build
    echo "✓ Docker images built"
fi

# Initialize database
read -p "Initialize database? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting database..."
    docker-compose up -d postgres
    sleep 5
    echo "Running migrations..."
    docker-compose run --rm translator-service alembic upgrade head
    echo "✓ Database initialized"
fi

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys:"
echo "   - Twilio credentials"
echo "   - STT provider API key (Deepgram/Whisper)"
echo "   - Translation provider API key (DeepL/OpenAI)"
echo "   - TTS provider API key (ElevenLabs/Azure)"
echo ""
echo "2. Configure Twilio:"
echo "   - Set webhook URL to: https://your-domain.com/voice/webhook"
echo "   - Enable Media Streams"
echo ""
echo "3. Start the service:"
echo "   docker-compose up -d"
echo ""
echo "4. Check service health:"
echo "   curl http://localhost:8000/health"
echo ""
echo "For more information, see README.md"
