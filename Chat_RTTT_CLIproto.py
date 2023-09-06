import json
import os
import openai
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
from pydub.playback import play
from charactr_api import CharactrAPISDK, Credentials
import time


# API-key
openai_api_key = os.environ['OPENAI_API_KEY']
charactr_client_key = os.environ['CHARACTR_CLIENT_KEY']
charactr_api_key = os.environ['CHARACTR_API_KEY']


openai.api_key = openai_api_key

credentials = Credentials(client_key=charactr_client_key, api_key=charactr_api_key)
charactr_api = CharactrAPISDK(credentials)

voice_id = 163
model = 'gpt-3.5-turbo'
parameters = {
    'temperature': 0.8,
    'max_tokens': 150,
    'top_p': 1,
    'presence_penalty': 0,
    'frequency_penalty': 0,
    'stop': None
}

conversation = [{'role': 'system', 'content': 'You are AI Assistant.'}]

def speech2text(audio_path: str) -> str:
    """Run a request to Whisper to convert speech to text."""
    try:
        start_time = time.time()
        with open(audio_path, 'rb') as audio_f:
            result = openai.Audio.transcribe('whisper-1', audio_f)
        end_time = time.time()
        whisper_time = end_time - start_time
        text = result['text']
    except Exception as e:
        raise Exception(e)
    return text, whisper_time

def update_conversation(request, conversation):
    start_time = time.time()
    user_request = {'role': 'user', 'content': request}
    conversation.append(user_request)
    result = openai.ChatCompletion.create(model=model, messages=conversation, **parameters)
    response = result['choices'][0]['message']['content'].strip()
    bot_response = {'role': 'assistant', 'content': response}
    conversation.append(bot_response)
    end_time = time.time()
    chat_gpt_time = end_time - start_time
    return chat_gpt_time

def record_audio():
    duration = int(input("How many seconds would you like to record? "))
    recording = sd.rec(int(duration * 44100), samplerate=44100, channels=1, dtype='int16')
    sd.wait()
    return np.array(recording, dtype='int16')

def main():
    choice = input("Would you like to use voice input or text input? (Enter 'v' or 't'): ").lower()
    if choice == 'v':
        print("Recording audio...")
        audio_data = record_audio()
        audio_segment = AudioSegment(audio_data.tobytes(), frame_rate=44100, sample_width=2, channels=1)
        audio_segment.export("recording.wav", format="wav")
        # Cnvert audio to text
        input_text, whisper_time = speech2text("recording.wav")
        print(f"Converted text from voice input: {input_text}")

    elif choice == 't':
        input_text = input("Please enter your text: ")
        whisper_time = 0  # Setting whisper_time to 0 for text input

    else:
        print("Invalid choice. Exiting.")
        return

    chat_gpt_time = update_conversation(input_text, conversation)

    start_time = time.time()
    tts_result = charactr_api.tts.convert(voice_id, conversation[-1]['content'])
    end_time = time.time()
    charactr_time = end_time - start_time

    with open('response.wav', 'wb') as f:
        f.write(tts_result['data'])

    response_audio = AudioSegment.from_wav('response.wav')
    play(response_audio)

    print(f"Time taken for Whisper transcription: {whisper_time:.2f} seconds")
    print(f"Time taken for ChatGPT response: {chat_gpt_time:.2f} seconds")
    print(f"Time taken for CharactrAPI response: {charactr_time:.2f} seconds")

    total_time = whisper_time + chat_gpt_time + charactr_time

    print(f"Total Time for response: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
