import whisper
import sounddevice as sd
import numpy as np
import soundfile as sf
import keyboard  

print(">> Press 'R' to start recording <<")

# Load whisper model 
model = whisper.load_model("base")

sample_rate = 16000
recording = False
audio_data = []   # store recorded chunks
stream = None     # audio stream object

def audio_callback(indata, frames, time, status):
    """Callback function to store audio while recording"""
    if recording:
        audio_data.append(indata.copy())

def start_recording():
    global stream, audio_data, recording
    audio_data = []   # reset buffer
    recording = True
    stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", callback=audio_callback)
    stream.start()
    print(">>> Recording Started (Press 'R' again to stop) <<<")

def stop_recording():
    global stream, recording, audio_data
    recording = False
    if stream:
        stream.stop()
        stream.close()
    print(">>> Recording Stopped. Transcribing... <<<")

    # Combine all chunks into one array
    if audio_data:
        audio_np = np.concatenate(audio_data, axis=0)
        sf.write("temp.wav", audio_np, sample_rate)

        # Transcribe with whisper
        result = model.transcribe("temp.wav")
        print("You said:", result['text'])

        with open("transcriptions.txt", "a", encoding="utf-8") as f:
            f.write(result['text'] + "\n")
    else:
        print("No audio captured.")

def toggle_recording():
    global recording
    while True:
        keyboard.wait("r")
        if not recording:
            start_recording()
        else:
            stop_recording()
            print(">> Press 'R' to start recording again <<")

toggle_recording()
