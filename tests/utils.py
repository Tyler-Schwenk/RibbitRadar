# tests/utils.py
import numpy as np
import wave
import os

def create_test_wav(filename, duration=1, freq=440, sample_rate=16000):
    # Create a sine wave and save it as a .wav file
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    sine_wave = 0.5 * np.sin(2 * np.pi * freq * t)

    # Write to a .wav file
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(np.int16(sine_wave * 32767).tobytes())
