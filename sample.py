from whisper_online import *
import librosa

src_lan = "zh"  # source language
tgt_lan = "zh"  # target language  -- same as source for ASR, "en" if translate task is used

asr = FasterWhisperASR(src_lan, "small")  # loads and wraps Whisper model
# set options:
# asr.set_translate_task()  # it will translate from lan into English
# asr.use_vad()  # set using VAD

online = OnlineASRProcessor(asr)  # create processing object with default buffer trimming option

whole_audo, _ = librosa.load("audio/backyard.mp3", sr=16000, dtype=np.float32)
for i in range(0, len(whole_audo), 16000):
	a = whole_audo[i:i+16000]
	online.insert_audio_chunk(a)
	beg_trans, end_trans, trans = online.process_iter()
	if beg_trans is not None:
		print(f"{beg_trans:.2f} {end_trans:.2f} {trans}") # do something with current partial output
	else:
		print("None")

# at the end of this audio processing
beg_trans, end_trans, trans = online.finish()
if beg_trans is not None:
	print(f"{beg_trans:.2f} {end_trans:.2f} {trans} finnish") # do something with current partial output
else:
	print("None finish")

online.init()  # refresh if you're going to re-use the object for the next audio