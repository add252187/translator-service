# Real-time Bidirectional Translation Service

A scalable Python system that acts as a real-time, bidirectional "simultaneous interpreter" for commercial phone calls between agents and clients speaking different languages.

## 🌟 Features

- **Real-time Translation**: Near-simultaneous bidirectional translation with <1.5s latency
- **Multi-language Support**: Clients can speak any supported language (English, French, German, Italian, Portuguese, Arabic, Galician, etc.)
- **Agent-focused**: Agents always speak and hear Spanish, regardless of client language
- **Automatic Language Detection**: Detects client language automatically from speech
- **Hyper-realistic Voices**: Uses ElevenLabs or similar neural TTS for natural-sounding speech
- **Scalable Architecture**: Supports 50+ concurrent calls
- **Pluggable Providers**: Easy to swap STT, Translation, and TTS providers
- **Production Ready**: Docker support, structured logging, error handling, and monitoring

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Twilio    │────▶│   WebSocket  │────▶│   Audio     │
│   Media     │     │   Handler    │     │   Router    │
│   Streams   │◀────│              │◀────│             │
└─────────────┘     └──────────────┘     └─────────────┘
                                                │
                            ┌───────────────────┼───────────────────┐
                            ▼                   ▼                   ▼
                    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
                    │     STT      │   │  Translation │   │     TTS      │
                    │  (Deepgram)  │   │   (DeepL/    │   │ (ElevenLabs) │
                    │              │   │    OpenAI)   │   │              │
                    └──────────────┘   └──────────────┘   └──────────────┘
```

### Call Flow

1. **Client speaks** (any language) → Audio captured by Twilio
2. **STT** converts speech to text and detects language
3. **Translation** converts client text to Spanish
4. **TTS** generates Spanish audio for agent
5. **Agent hears** Spanish translation

And simultaneously in reverse:

1. **Agent speaks** Spanish → Audio captured
2. **STT** converts Spanish speech to text
3. **Translation** converts to client's detected language
4. **TTS** generates audio in client's language
5. **Client hears** in their own language

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose (optional but recommended)
- Twilio account with Programmable Voice
- API keys for:
  - STT provider (Deepgram/Whisper/ElevenLabs)
  - Translation provider (DeepL/OpenAI)
  - TTS provider (ElevenLabs/Azure)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd translator-service
```

2. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

3. **Install dependencies** (if running locally)
```bash
pip install -r requirements.txt
```

### Running with Docker (Recommended)

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f translator-service

# Stop services
docker-compose down
```

### Running Locally

```bash
# Run database migrations (if using PostgreSQL)
alembic upgrade head

# Start the service
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

```env
# Twilio Configuration
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
WEBHOOK_BASE_URL=https://your-domain.com

# Provider Selection
STT_PROVIDER=deepgram  # Options: deepgram, whisper, elevenlabs
TRANSLATION_PROVIDER=deepl  # Options: deepl, openai
TTS_PROVIDER=elevenlabs  # Options: elevenlabs, azure

# API Keys
DEEPGRAM_API_KEY=your_key
DEEPL_API_KEY=your_key
ELEVENLABS_TTS_API_KEY=your_key

# Performance
MAX_CONCURRENT_CALLS=50
AUDIO_CHUNK_DURATION_MS=500
VAD_AGGRESSIVENESS=2  # 0-3

# Language Settings
DEFAULT_AGENT_LANGUAGE=es
BYPASS_TRANSLATION_FOR_SAME_LANGUAGE=false
```

### Twilio Configuration

1. **Create a Twilio Phone Number**
2. **Configure Voice Webhook**:
   - URL: `https://your-domain.com/voice/webhook`
   - Method: POST
3. **Enable Media Streams** in your Twilio console

### SSL/TLS Setup

For production, configure SSL certificates:

1. Place certificates in `./ssl/` directory
2. Update `nginx.conf` with your domain
3. Uncomment SSL configuration in `nginx.conf`

## 📡 API Endpoints

### HTTP Endpoints

- `GET /` - Service information
- `GET /health` - Health check
- `POST /voice/webhook` - Twilio voice webhook (TwiML)
- `GET /calls` - List active calls
- `GET /calls/{call_sid}` - Get call details
- `POST /calls/{call_sid}/end` - End a call
- `GET /metrics` - System metrics
- `POST /test/translate` - Test translation

### WebSocket Endpoints

- `WS /ws/media-stream` - Twilio Media Streams connection

## 🧪 Testing

### Run Unit Tests

```bash
pytest tests/ -v
```

### Test Translation Endpoint

```bash
curl -X POST "http://localhost:8000/test/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you?",
    "source_language": "en",
    "target_language": "es"
  }'
```

### Simulate a Call (Development)

```python
# See tests/test_basic_flow.py for examples
```

## 📊 Monitoring

### Metrics Available

- Active sessions count
- Average latency (ms)
- STT/Translation/TTS request counts
- Error rates
- Language detection confidence

### Logging

Structured JSON logs include:
- Call ID
- Request ID
- Detected language
- Component latencies
- Error details

### Health Checks

```bash
curl http://localhost:8000/health
```

## 🔌 Extending Providers

### Adding a New STT Provider

1. Create a new file in `app/stt/`
2. Inherit from `STTProvider` base class
3. Implement required methods
4. Register in `app/stt/factory.py`

Example:
```python
class MySTTProvider(STTProvider):
    async def transcribe(self, audio_data, sample_rate, **kwargs):
        # Your implementation
        pass
```

### Adding a New Translation Provider

Similar process in `app/translation/` directory.

### Adding a New TTS Provider

Similar process in `app/tts/` directory.

## 🏭 Production Deployment

### Scaling Considerations

1. **Horizontal Scaling**: Deploy multiple instances behind a load balancer
2. **Database**: Use PostgreSQL for production (not SQLite)
3. **Redis**: Add Redis for caching and session management
4. **Monitoring**: Integrate with Prometheus/Grafana
5. **Logging**: Ship logs to ELK stack or similar

### Performance Tuning

- Adjust `AUDIO_CHUNK_DURATION_MS` for latency vs quality tradeoff
- Tune `VAD_AGGRESSIVENESS` based on environment noise
- Configure `MAX_CONCURRENT_CALLS` based on server resources
- Use connection pooling for database and external APIs

### Security Best Practices

1. Always use HTTPS/WSS in production
2. Rotate API keys regularly
3. Implement rate limiting
4. Use environment-specific configurations
5. Never log conversation content in production
6. Implement proper authentication for admin endpoints

## 🐛 Troubleshooting

### Common Issues

1. **High Latency**
   - Check network connectivity to API providers
   - Reduce audio chunk size
   - Ensure sufficient server resources

2. **Language Detection Failures**
   - Increase initial audio buffer size
   - Adjust confidence threshold
   - Provide language hints when possible

3. **Audio Quality Issues**
   - Verify sample rates match throughout pipeline
   - Check audio format conversions
   - Ensure proper codec support

4. **WebSocket Disconnections**
   - Check firewall/proxy timeout settings
   - Implement reconnection logic
   - Monitor network stability

## 📝 License

[Your License Here]

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## 📧 Support

For issues and questions:
- GitHub Issues: [Link]
- Email: support@example.com
- Documentation: [Link]

## 🙏 Acknowledgments

- Twilio for telephony infrastructure
- Deepgram for speech recognition
- DeepL for translation
- ElevenLabs for text-to-speech
- FastAPI for the web framework
