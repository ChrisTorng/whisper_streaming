import sys
import time
import msvcrt
import threading         # 新增 threading
import numpy as np
import wave
import pyaudio
from whisper_online import *

print("stderr", file=sys.stderr)

last_time = time.time()
def printt(text):
    global last_time
    print(f"{(time.time() - last_time):.3f} {text}")
    last_time = time.time()

# Parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # 8000, 16000, 32000
WAIT_SECONDS = 1
FRAMES_PER_BUFFER = int(RATE * WAIT_SECONDS)  # 320

src_lan = "zh"  # source language
tgt_lan = "zh"  # target language -- same as source for ASR, "en" if translate task is used
# class LogArgs:
# 	log_level = logging.WARNING

# logger = logging.getLogger(__name__)
# set_logging(LogArgs(), logger)

# 新增：定義 process_iter() 執行狀態與保護 lock
processing_lock = threading.Lock()
processing_busy = False

def try_process():
    global processing_busy
    with processing_lock:
        if processing_busy:
            return
        processing_busy = True
    # 執行 process_iter() 呼叫
    beg_trans, end_trans, trans = online.process_iter()
    if beg_trans is not None:
        print(f"{beg_trans:.2f} {end_trans:.2f} {trans}")
    else:
        print("None")
    with processing_lock:
        processing_busy = False

printt("loading")
asr = FasterWhisperASR(src_lan, "turbo")  # loads and wraps Whisper model
# set options:
# asr.set_translate_task()  # it will translate from lan into English
asr.use_vad()  # set using VAD 

# online = OnlineASRProcessor(asr, create_tokenizer(tgt_lan))  # create processing object
online = OnlineASRProcessor(asr)  # create processing object
# online = VACOnlineASRProcessor(WAIT_SECONDS, asr)  # create processing object

pa = pyaudio.PyAudio()
stream = pa.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=FRAMES_PER_BUFFER)
printt("listening")

# # 新增：累積錄音區塊的列表
# recorded_audio = []
# # 新增：累積放大後(經 normalization)的錄音區塊
# normalized_audio = []

while True:   # processing loop:
    audio_frames = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=True)
	# printt(f"audio chunk received: {len(audio_frames)} {audio_frames[:10]}")
    # printt(f"audio chunk received: {len(audio_frames)}")
    
    # 將原始錄音轉換為 numpy 陣列後累積(注意：pyaudio 回傳 bytes)
    raw_np = np.frombuffer(audio_frames, dtype=np.int16)
    # recorded_audio.append(raw_np)

    pcm_buffer = bytearray()
    pcm_buffer.extend(audio_frames)
    audio_array = (
        np.frombuffer(pcm_buffer, dtype=np.int16).astype(np.float32)
        / 32768.0
    )
    
    # normalization: 若當前音量低於 threshold，則放大增益
    # threshold = 0.5  # 可依需求調整
    # current_max = np.max(np.abs(audio_array))
    # if 0 < current_max < threshold:
    #     gain = threshold / current_max
    #     audio_array *= gain
    #     printt(f"inserting audio chunk {len(audio_array)} *{gain}")

    audio_array *= 25
    # # 將 normalization 後的 audio_array 轉回 int16 並儲存
    # normalized_audio.append((audio_array * 32768).astype(np.int16))
    
    # printt(f"inserting audio chunk {len(audio_array)} {audio_array[:10]}")
    # printt(f"inserting audio chunk {len(audio_array)} *{25}")
    online.insert_audio_chunk(audio_array)
    # 呼叫 process_iter()，但若前一次尚未結束則不重覆執行
    threading.Thread(target=try_process).start()

    # 結束錄音判斷：利用 msvcrt 檢查是否有鍵盤輸入
    if msvcrt.kbhit():
        key = msvcrt.getch()
        if key == b'\x1b':  # Esc 鍵
            printt("Esc pressed, stopping recording")
            break

# # 結束錄音後：存檔
# wav_data = np.concatenate(recorded_audio, axis=0)
# wav_file_path = "audio/whisper_streaming.wav"
# with wave.open(wav_file_path, 'wb') as wf:
#     wf.setnchannels(CHANNELS)
#     wf.setsampwidth(2)  # int16 -> 2 bytes
#     wf.setframerate(RATE)
#     wf.writeframes(wav_data.tobytes())
# printt(f"錄音存檔至 {wav_file_path}")

# wav_normalized_data = np.concatenate(normalized_audio, axis=0)
# wav_normalized_file_path = "audio/whisper_streaming_normalized.wav"
# with wave.open(wav_normalized_file_path, 'wb') as wf_norm:
#     wf_norm.setnchannels(CHANNELS)
#     wf_norm.setsampwidth(2)
#     wf_norm.setframerate(RATE)
#     wf_norm.writeframes(wav_normalized_data.tobytes())
# printt(f"放大後錄音存檔至 {wav_normalized_file_path}")

# at the end of this audio processing
beg_trans, end_trans, trans = online.finish()
if beg_trans is not None:
    print(f"{beg_trans:.2f} {end_trans:.2f} {trans} finish")  # do something with current partial output
else:
	print("None finish")

online.init()  # refresh if you're going to re-use the object for the next audio