@echo off
REM Setup script for the Real-time Bidirectional Translation Service (Windows)

echo =========================================
echo Real-time Translation Service Setup
echo =========================================

REM Check Python version
echo Checking Python version...
python --version 2>&1 | findstr /R "3\.1[1-9]" >nul
if errorlevel 1 (
    echo Error: Python 3.11+ is required
    exit /b 1
)
echo OK: Python 3.11+ found

REM Check for Docker
echo Checking for Docker...
docker --version >nul 2>&1
if errorlevel 1 (
    echo Error: Docker is not installed
    exit /b 1
)
echo OK: Docker installed

REM Create .env file if it doesn't exist
if not exist .env (
    echo Creating .env file...
    copy .env.example .env
    echo OK: .env file created
    echo.
    echo IMPORTANT: Edit .env file with your API keys before running the service
) else (
    echo OK: .env file exists
)

REM Create necessary directories
echo Creating directories...
if not exist data mkdir data
if not exist logs mkdir logs
if not exist ssl mkdir ssl
if not exist alembic\versions mkdir alembic\versions
echo OK: Directories created

REM Install Python dependencies
set /p install_deps="Install Python dependencies locally? (y/n): "
if /i "%install_deps%"=="y" (
    echo Installing Python dependencies...
    pip install -r requirements.txt
    echo OK: Dependencies installed
)

REM Build Docker images
set /p build_docker="Build Docker images? (y/n): "
if /i "%build_docker%"=="y" (
    echo Building Docker images...
    docker-compose build
    echo OK: Docker images built
)

REM Initialize database
set /p init_db="Initialize database? (y/n): "
if /i "%init_db%"=="y" (
    echo Starting database...
    docker-compose up -d postgres
    timeout /t 5 /nobreak >nul
    echo Running migrations...
    docker-compose run --rm translator-service alembic upgrade head
    echo OK: Database initialized
)

echo.
echo =========================================
echo Setup Complete!
echo =========================================
echo.
echo Next steps:
echo 1. Edit .env file with your API keys:
echo    - Twilio credentials
echo    - STT provider API key (Deepgram/Whisper)
echo    - Translation provider API key (DeepL/OpenAI)
echo    - TTS provider API key (ElevenLabs/Azure)
echo.
echo 2. Configure Twilio:
echo    - Set webhook URL to: https://your-domain.com/voice/webhook
echo    - Enable Media Streams
echo.
echo 3. Start the service:
echo    docker-compose up -d
echo.
echo 4. Check service health:
echo    curl http://localhost:8000/health
echo.
echo For more information, see README.md
pause
