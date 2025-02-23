import sys
import threading
import numpy as np
import wave
import sounddevice as sd
from keyboard_handler import KeyboardHandler
from whisper_online import *

print("stderr", file=sys.stderr)

last_time = time.time()
def printt(text):
    global last_time
    print(f"{(time.time() - last_time):.3f} {text}")
    last_time = time.time()

# Parameters
CHANNELS = 2  # 修改為 2 聲道
RATE = 16000 # 8000, 16000, 32000
WAIT_SECONDS = 1
FRAMES_PER_BUFFER = int(RATE * WAIT_SECONDS) # 320

src_lan = "zh"  # source language
tgt_lan = "zh"  # target language  -- same as source for ASR, "en" if translate task is used

# 為兩個聲道分別建立處理狀態與鎖
processing_lock_left = threading.Lock()
processing_busy_left = False
processing_lock_right = threading.Lock()
processing_busy_right = False

def try_process_channel(online_processor, lock, busy_flag, channel_name):
    with lock:
        if busy_flag[0]:
            return
        busy_flag[0] = True
    beg_trans, end_trans, trans = online_processor.process_iter()
    if beg_trans is not None:
        print(f"{channel_name}: {beg_trans:.2f} {end_trans:.2f} {trans}")
    else:
        print(f"{channel_name}: None")
    with lock:
        busy_flag[0] = False

printt("loading")
asr = FasterWhisperASR(src_lan, "turbo")  # loads and wraps Whisper model

# 為兩個聲道建立獨立的處理器
online_left = OnlineASRProcessor(asr)
online_right = OnlineASRProcessor(asr)

keyboard = KeyboardHandler()
keyboard.init()

with sd.InputStream(samplerate=RATE, channels=CHANNELS, dtype='int16') as stream:
    printt("listening")

    while True:   # processing loop:
        audio_frames, overflowed = stream.read(FRAMES_PER_BUFFER)
        if overflowed:
            printt(f"audio chunk received: {len(audio_frames)} {overflowed}")

        pcm_buffer = bytearray()
        pcm_buffer.extend(audio_frames)
        audio_array = (
            np.frombuffer(pcm_buffer, dtype=np.int16).astype(np.float32)
            / 32768.0
        )

        # 分離左右聲道
        audio_left = audio_array[::2]
        audio_right = audio_array[1::2]

        # 放大音量
        audio_left *= 25
        audio_right *= 25

        # 分別處理兩個聲道
        online_left.insert_audio_chunk(audio_left)
        online_right.insert_audio_chunk(audio_right)

        # 為兩個聲道分別建立處理緒
        busy_left = [processing_busy_left]
        busy_right = [processing_busy_right]
        threading.Thread(
            target=try_process_channel,
            args=(online_left, processing_lock_left, busy_left, "左聲道")
        ).start()
        threading.Thread(
            target=try_process_channel,
            args=(online_right, processing_lock_right, busy_right, "右聲道")
        ).start()

        # 使用新的鍵盤檢查機制
        if keyboard.check_key():
            printt("Esc pressed, stopping recording")
            break

# 還原鍵盤設定
keyboard.restore()

# 處理最後的音訊片段
print("Left channel final result:")
beg_trans, end_trans, trans = online_left.finish()
if beg_trans is not None:
    print(f"左聲道: {beg_trans:.2f} {end_trans:.2f} {trans}")
else:
    print("左聲道: None")

print("Right channel final result:")
beg_trans, end_trans, trans = online_right.finish()
if beg_trans is not None:
    print(f"右聲道: {beg_trans:.2f} {end_trans:.2f} {trans}")
else:
    print("右聲道: None")

# 重置處理器
online_left.init()
online_right.init()