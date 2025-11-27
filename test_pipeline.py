import asyncio
import struct
import math

async def test_pipeline():
    # Test Translation
    print('=== Testing DeepL Translation ===')
    from app.translation import get_translation_provider
    translator = get_translation_provider()
    await translator.initialize()
    
    result = await translator.translate('Hello, how are you?', target_language='es', source_language='en')
    print(f'Translation: "{result.translated_text}"')
    
    await translator.close()
    
    # Test TTS
    print('\n=== Testing ElevenLabs TTS ===')
    from app.tts import get_tts_provider
    tts = get_tts_provider()
    await tts.initialize()
    
    result = await tts.synthesize('Hola, como estas?', language='es')
    print(f'TTS Result: {len(result.audio_data) if result.audio_data else 0} bytes')
    
    if result.audio_data:
        # Save to file for testing
        with open('test_output.mp3', 'wb') as f:
            f.write(result.audio_data)
        print('Audio saved to test_output.mp3')
    
    await tts.close()
    
    print('\n=== All tests completed ===')

if __name__ == "__main__":
    asyncio.run(test_pipeline())
