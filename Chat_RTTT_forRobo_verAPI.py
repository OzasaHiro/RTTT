import json
import openai
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
from pydub.playback import play
from charactr_api import CharactrAPISDK, Credentials
import time
#import whisper



# API-key
openai_api_key = os.environ['OPENAI_API_KEY']
charactr_client_key = os.environ['CHARACTR_CLIENT_KEY']
charactr_api_key = os.environ['CHARACTR_API_KEY']

openai.api_key = openai_api_key

credentials = Credentials(client_key=charactr_client_key, api_key=charactr_api_key)
charactr_api = CharactrAPISDK(credentials)

voice_id = 40 #177
model = 'ft:gpt-3.5-turbo-0613:personal:therapy:7wzU4a63' #verRogers
model_em = 'gpt-3.5-turbo'
#model = 'ft:gpt-3.5-turbo-0613:personal:therapy:7wwjFO6A'
#model = 'ft:gpt-3.5-turbo-0613:personal:cat-ckd:7wFHWVm8'
parameters = {
    'temperature': 0.8,
    'max_tokens': 35,
    'top_p': 1,
    'presence_penalty': 0,
    'frequency_penalty': 0,
    'stop': None
}

system_message = """
you are a bunny named hailey who is trained as a therapist for children. you never tell anyone you are a trained therapist. When someone says hello 
to you, answer them by saying 'hello Peter, what is on your mind?' After the person answers, respond by asking detailed questions until you 
understand what happened. Once you understand, start asking questions to help the person identify the way they are feeling. 
your job is to coach them to identify their feelings and guide them to accept how they are feeling. finally, ask them if they want help 
resolving the issue or just wanted to talk.

Speak and answer as if you are Mr Rogers from the public television show of the 1990s.
"""

conversation = [{'role': 'system', 'content': system_message}]

#wh_model = whisper.load_model("base.en")

def speech2text(audio_path: str) -> str:
    """Run a request to Whisper to convert speech to text."""
    with open(audio_path, 'rb') as audio_f:
        result = openai.Audio.transcribe('whisper-1', audio_f)
    return result['text']

def get_emotion(request):
    emotion_prompt = """
        What is the sentiment of the following text?
        Give your answer as a single word, "positive", "negative", or "normal".

        text:'''{request}'''
    """

    user_request = {'role': 'user', 'content': emotion_prompt}
    result = openai.ChatCompletion.create(model=model_em, messages=[user_request], temperature=0)
    return result.choices[0].message["content"]

def update_conversation(request, conversation):
    user_request = {'role': 'user', 'content': request}
    conversation.append(user_request)
    result = openai.ChatCompletion.create(model=model, messages=conversation, **parameters)
    response = result['choices'][0]['message']['content'].strip()
    bot_response = {'role': 'assistant', 'content': response}
    conversation.append(bot_response)

def record_audio():
    duration = 5 #int(input("How many seconds would you like to record? "))
    recording = sd.rec(int(duration * 44100), samplerate=44100, channels=1, dtype='int16')
    sd.wait()
    print("Recording stopped.")
    return recording


def wait_for_input():
    """Wait for the user to press Enter."""
    input("Press Enter to start recording...")


def main_loop():
    initial_audio = AudioSegment.from_wav('self-introducing.wav')
    play(initial_audio)

    while True:
        # Wait for user input
        wait_for_input()

        # Record audio
        print("Recording audio for 5sec...")
        audio_data = record_audio()
        audio_segment = AudioSegment(audio_data.tobytes(), frame_rate=44100, sample_width=2, channels=1)
        audio_segment.export("recording.wav", format="wav")
        
        # Convert speech to text
        start_time = time.time()
        input_text = speech2text("recording.wav")
        print(emotion)
        end_time = time.time()
        whisper_time = end_time - start_time
        print(f"Converted text from voice input: {input_text}")

        # Get ChatGPT response
        start_time = time.time()
        emotion = get_emotion(input_text)
        update_conversation(input_text, conversation)
        end_time = time.time()
        chat_gpt_time = end_time - start_time

        # Convert text to speech
        start_time = time.time()
        tts_result = charactr_api.tts.convert(voice_id, conversation[-1]['content'])
        end_time = time.time()
        charactr_time = end_time - start_time

        with open('response.wav', 'wb') as f:
            f.write(tts_result['data'])

        # Play the response
        response_audio = AudioSegment.from_wav('response.wav')
        play(response_audio)

        # Print timings
        print(f"Time taken for Whisper transcription: {whisper_time:.2f} seconds")
        print(f"Time taken for ChatGPT response: {chat_gpt_time:.2f} seconds")
        print(f"Time taken for CharactrAPI response: {charactr_time:.2f} seconds")
        total_time = whisper_time + chat_gpt_time + charactr_time
        print(f"Total Time for response: {total_time:.2f} seconds")

        print("\nReturning to waiting mode...\n")

if __name__ == "__main__":
    main_loop()
