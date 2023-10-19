import sys
import sounddevice as sd
import wave
from whisper_online import *

print("stderr", file=sys.stderr)

last_time = time.time()
def printt(text):
	global last_time
	print(f"{(time.time() - last_time):.3f} {text}")
	last_time = time.time()

# Parameters
RATE = 16000 # 8000, 16000, 32000
FRAMES_PER_BUFFER = int(RATE * 3) # 320

src_lan = "zh"  # source language
tgt_lan = "zh"  # target language  -- same as source for ASR, "en" if translate task is used

printt("loading")
asr = FasterWhisperASR(src_lan, "small")  # loads and wraps Whisper model
# set options:
# asr.set_translate_task()  # it will translate from lan into English
asr.use_vad()  # set using VAD 

online = OnlineASRProcessor(asr, create_tokenizer(tgt_lan))  # create processing object

with sd.InputStream(samplerate=RATE, channels=1, dtype='int16') as stream:
	printt("listening")

	while True:   # processing loop:
		audio_data, overflowed = stream.read(FRAMES_PER_BUFFER)
		printt("audio chunk received")

		audio_recorded_filename = f'audio/temp/RECORDED-{str(time.time())}.wav'
		wf = wave.open(audio_recorded_filename, 'wb')
		wf.setnchannels(1)
		wf.setsampwidth(2)
		wf.setframerate(RATE)
		wf.writeframes(audio_data)
		wf.close()
		printt("saved")

		online.insert_audio_chunk(audio_data.astype(np.float32))
		o = online.process_iter()
		print(o) # do something with current partial output

# at the end of this audio processing
o = online.finish()
print(o)  # do something with the last output

online.init()  # refresh if you're going to re-use the object for the next audio