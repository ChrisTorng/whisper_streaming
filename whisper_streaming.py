import pyaudio
import wave
# import io
from whisper_online import *

print("stderr", file=sys.stderr)

# Delete all audio files in temp folder
import os
import glob
files = glob.glob('audio/temp/*.wav')
for f in files:
    os.remove(f)

last_time = time.time()
def printt(text):
	global last_time
	print(f"{(time.time() - last_time):.3f} {text}")
	last_time = time.time()

# Parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000 # 8000, 16000, 32000
FRAMES_PER_BUFFER = int(RATE * 0.03) # 30ms=480, max supported tried

src_lan = "zh"  # source language
tgt_lan = "zh"  # target language  -- same as source for ASR, "en" if translate task is used

printt("loading")
asr = FasterWhisperASR(src_lan, "small")  # loads and wraps Whisper model
# set options:
# asr.set_translate_task()  # it will translate from lan into English
asr.use_vad()  # set using VAD 

online = OnlineASRProcessor(asr, create_tokenizer(tgt_lan))  # create processing object

pa = pyaudio.PyAudio()
stream = pa.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=FRAMES_PER_BUFFER)
printt("listening")

while True:   # processing loop:
	audio_frames = stream.read(FRAMES_PER_BUFFER)
	printt("audio chunk received")

	# audio_buffer = io.BytesIO()
	# wf = wave.open(audio_buffer, 'wb')
	# wf.setnchannels(CHANNELS)
	# wf.setsampwidth(pa.get_sample_size(FORMAT))
	# wf.setframerate(RATE)
	# wf.writeframes(audio_frames)
	# wf.close()
	# textlist = asr.transcribe(wf, init_prompt="繁體中文台灣用語")

	# audio_recorded_filename = f'audio/temp/RECORDED-{str(time.time())}.wav'
	# wf = wave.open(audio_recorded_filename, 'wb')
	# wf.setnchannels(CHANNELS)
	# wf.setsampwidth(pa.get_sample_size(FORMAT))
	# wf.setframerate(RATE)
	# wf.writeframes(audio_frames)
	# wf.close()
	# printt("saved")
	# textlist = asr.transcribe(audio_recorded_filename, init_prompt="繁體中文台灣用語")

	# audio_data = np.ndarray(buffer=audio_frames, dtype=np.int16, shape=(FRAMES_PER_BUFFER, ))
	# audio_data = np.frombuffer(audio_frames, dtype=np.int16)
	# textlist = asr.transcribe(audio_data, init_prompt="繁體中文台灣用語")

	# audio_stream = io.BytesIO(audio_frames)
	# textlist = asr.transcribe(audio_stream, init_prompt="繁體中文台灣用語")
	
	# for segment in textlist:
	# 	printt(segment.text)
	# print(textlist)

	audio_array = np.frombuffer(audio_frames, dtype=np.float32)
	print(type(audio_array),file=sys.stderr)
	print(audio_array.dtype,file=sys.stderr)
	print(audio_array.shape,file=sys.stderr)

	online.insert_audio_chunk(audio_array)
	o = online.process_iter()
	print(o) # do something with current partial output

# at the end of this audio processing
o = online.finish()
print(o)  # do something with the last output

online.init()  # refresh if you're going to re-use the object for the next audio