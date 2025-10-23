import pyaudio
import sys
import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

CHUCK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1 if sys.platform == 'darwin' else 2
r = 16000
duration = 5
WAVE_OUTPUT_FILENAME = "test.wav"

with wave.open('test.wav', 'wb') as wf:
    p = pyaudio.PyAudio()
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(r)

    stream = p.open(format=FORMAT, channels=CHANNELS, rate=r, input=True)

    print("[+] Recording")
    for _ in range(0, r // CHUCK * duration):
        data = stream.read(CHUCK)
        wf.writeframes(data)

    print("[+] Done recording")
    stream.close()
    p.terminate()

print(f"[+] Audio recorded and saved to {WAVE_OUTPUT_FILENAME}")


rate, data = wavfile.read(WAVE_OUTPUT_FILENAME)