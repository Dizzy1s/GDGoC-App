# could have problems with surround sounds

import pyaudio
import wave
import threading
import time

import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.environ["GOOGLE_API_KEY"]
AUDIO_FILE_PATH = "./audio_input.wav"

if not GEMINI_API_KEY:
    print("Gemini Api key not provided!\n")
    quit()

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("models/gemini-2.0-flash")

filename = AUDIO_FILE_PATH
duration = 5
sample_rate = 44100
chunk = 1024
channels = 1
format = pyaudio.paInt16
audio = pyaudio.PyAudio()

frames = []

def record(stop_event):
    global frames

    stream = audio.open(format=format,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk)
    
    while not stop_event.is_set():
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(format))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

    # print(f"Saved to {filename}")

    frames = []

# used to take input from user
def take_audio_input():
    stop_event = threading.Event()
    
    input("Press Enter to start recording...") 

    t = threading.Thread(target=record, args=(stop_event,))
    t.start()
    
    input("Press Enter to stop recording...")
    stop_event.set()
    t.join()
    
    print("Recording stopped.")

    return AUDIO_FILE_PATH

# used for response from gemini
def get_response(audio_path):
    uploaded_file = genai.upload_file(audio_path)
    response = model.generate_content([
        uploaded_file,
        "Write the exact words used in the audio"
    ])
    return response.text.strip()

def get_emotion(audio_path):
    uploaded_file = genai.upload_file(audio_path)
    prompt = """
Listen to the tone, pace, and expression in this voice clip.
What emotion is the speaker likely feeling?

Following emotions are detected:
1) ...
2) ...
"""
    response = model.generate_content([uploaded_file, prompt])
    return response.text.strip()

# below is for testing
while True:
    audio_file_path = take_audio_input()

    words_spoken = get_response(audio_file_path)
    print(f"Words spoken: {words_spoken}")