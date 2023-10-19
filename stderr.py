import sys
import time
import wave
import sounddevice as sd

print('stderr', file=sys.stderr)

RATE = 16000 # 8000, 16000, 32000
FRAMES_PER_BUFFER = int(RATE * 3) # 320
print("Listening")

with sd.InputStream(samplerate=RATE, channels=1, dtype='int16') as stream:
    audio_frames = stream.read(FRAMES_PER_BUFFER)
    print("audio chunk received")

    audio_recorded_filename = f'audio/temp/RECORDED-{str(time.time())}.wav'
    wf = wave.open(audio_recorded_filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(RATE)
    wf.writeframes(audio_frames[0].tobytes())
    wf.close()