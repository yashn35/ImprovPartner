# STREAMING CODE 

import asyncio
import signal
import openai
from pydantic_settings import BaseSettings, SettingsConfigDict
from vocode.helpers import create_streaming_microphone_input_and_speaker_output
from vocode.logging import configure_pretty_logging
from vocode.streaming.agent.chat_gpt_agent import ChatGPTAgent
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.transcriber import (
    DeepgramTranscriberConfig,
    PunctuationEndpointingConfig,
)
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramTranscriber
from vocode.streaming.synthesizer.cartesia_synthesizer import CartesiaSynthesizer
from vocode.streaming.models.synthesizer import CartesiaSynthesizerConfig
from config import openai_api_key, deepgram_api_key, cartesia_api_key

from vocode.streaming.models.synthesizer import CartesiaVoiceControls

configure_pretty_logging()

class Settings(BaseSettings):
    openai_api_key: str = openai_api_key
    deepgram_api_key: str = deepgram_api_key
    cartesia_api_key: str = cartesia_api_key

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()

def generate_scene():
    openai.api_key = settings.openai_api_key
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a creative improv scene generator."},
            {"role": "user", "content": "Generate a scene description and genre for an improv game. FOR NOW ONLY DO 'Romantic Comedy' for the genre"}
        ]
    )
    scene = response.choices[0].message.content
    return scene

async def main():
    scene = generate_scene()
    print(f"{scene}")
    input("Press Enter to start the scene...")

    (
        microphone_input,
        speaker_output,
    ) = create_streaming_microphone_input_and_speaker_output(
        use_default_devices=False,
    )

    conversation = StreamingConversation(
        output_device=speaker_output,
        transcriber=DeepgramTranscriber(
            DeepgramTranscriberConfig.from_input_device(
                microphone_input,
                endpointing_config=PunctuationEndpointingConfig(),
                api_key=settings.deepgram_api_key,
            ),
        ),
        agent=ChatGPTAgent(
            ChatGPTAgentConfig(
                openai_api_key=settings.openai_api_key,
                initial_message=BaseMessage(text=f"Let's start the scene! The scene is about {scene}"),
                prompt_preamble=f"""You are an AI improv partner. The scene is: {scene}. 
                Respond to the user's inputs in character, keeping the scene and genre in mind. 
                Keep your responses concise and natural for an improv scene.""",
            )
        ),
        synthesizer=CartesiaSynthesizer(
            CartesiaSynthesizerConfig.from_output_device(
                speaker_output,
                api_key=settings.cartesia_api_key,
                model_id='sonic-english',
                voice_id='79a125e8-cd45-4c13-8a67-188112f4dd22', # https://play.cartesia.ai/voices/79a125e8-cd45-4c13-8a67-188112f4dd22
            )
        ),
    )
    
    await conversation.start()
    print("Scene started. Speak your lines, and press Ctrl+C to end.")
    signal.signal(signal.SIGINT, lambda _0, _1: asyncio.create_task(conversation.terminate()))
    
    while conversation.is_active():
        chunk = await microphone_input.get_audio()
        conversation.receive_audio(chunk)

if __name__ == "__main__":
    asyncio.run(main())