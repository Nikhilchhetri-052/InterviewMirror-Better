import whisper
import sounddevice as sd
import numpy as np
import queue
import threading
import soundfile as sf
from datetime import datetime

# Load whisper model
model = whisper.load_model("base")  

sample_rate = 16000
duration = 10  # seconds for each chunk

q = queue.Queue()

def record_audio():
    print("Recording...")
    while True:
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        q.put(np.squeeze(audio))

def transcribe_audio():
    while True:
        audio = q.get()
        print("Transcribing...")
        # Save audio to a temp file
        sf.write("temp.wav", audio, sample_rate)
        # Transcribe
        result = model.transcribe("temp.wav")
        print("You said:", result['text'])
        with open("transcriptions.txt", "a", encoding="utf-8") as f:
            f.write(result['text'] + "\n")
        
# Start recording in a separate thread
threading.Thread(target=record_audio, daemon=True).start()
# Start transcription loop
transcribe_audio()
