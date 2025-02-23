import sys
import threading
import numpy as np
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
CHANNELS = 1
RATE = 16000 # 8000, 16000, 32000
WAIT_SECONDS = 1
FRAMES_PER_BUFFER = int(RATE * WAIT_SECONDS) # 320

src_lan = "zh"  # source language
tgt_lan = "zh"  # target language  -- same as source for ASR, "en" if translate task is used

# 定義 process_iter() 執行狀態與 lock
processing_lock = threading.Lock()
processing_busy = False

def try_process():
    global processing_busy
    with processing_lock:
        if processing_busy:
            return
        processing_busy = True
    beg_trans, end_trans, trans = online.process_iter()
    if beg_trans is not None:
        print(f"{beg_trans:.2f} {end_trans:.2f} {trans}")
    else:
        print("None")
    with processing_lock:
        processing_busy = False

printt("loading")
asr = FasterWhisperASR(src_lan, "turbo")  # loads and wraps Whisper model

# online = OnlineASRProcessor(asr, create_tokenizer(tgt_lan))  # create processing object
online = OnlineASRProcessor(asr)  # create processing object

# # 累積錄音區塊的列表
# recorded_audio = []

# # 累積放大後(經 normalization)的錄音區塊
# normalized_audio = []

keyboard = KeyboardHandler()
keyboard.init()

with sd.InputStream(samplerate=RATE, channels=CHANNELS, dtype='int16') as stream:
    printt("listening")

    while True:   # processing loop:
        audio_frames, overflowed = stream.read(FRAMES_PER_BUFFER)
        # 累積錄音數據
        # recorded_audio.append(audio_frames.copy())
        # printt(f"audio chunk received: {len(audio_frames)} {overflowed} {audio_frames[:10]}")
        if overflowed:
            printt(f"audio chunk received: {len(audio_frames)} {overflowed}")

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
        # 將 normalization 後的 audio_array 轉回 int16 並儲存
        # normalized_audio.append((audio_array * 32768).astype(np.int16))

        # printt(f"inserting audio chunk {len(audio_array)} {audio_array[:10]}")
        # printt(f"inserting audio chunk {len(audio_array)} *{25}")
        online.insert_audio_chunk(audio_array)
        # 呼叫 process_iter()，若前一次尚未完成則略過
        threading.Thread(target=try_process).start()

        # 使用新的鍵盤檢查機制
        if keyboard.check_key():
            printt("Esc pressed, stopping recording")
            break

# 還原鍵盤設定
keyboard.restore()

# # 結束錄音後：存檔
# wav_data = np.concatenate(recorded_audio, axis=0)
# wav_file_path = "audio/whisper_streaming_sd.wav"
# with wave.open(wav_file_path, 'wb') as wf:
#     wf.setnchannels(CHANNELS)
#     wf.setsampwidth(2)  # int16 -> 2 bytes
#     wf.setframerate(RATE)
#     wf.writeframes(wav_data.tobytes())
# printt(f"錄音存檔至 {wav_file_path}")

# # 存檔 normalization 後的音訊
# wav_normalized_data = np.concatenate(normalized_audio, axis=0)
# wav_normalized_file_path = "audio/whisper_streaming_sd_normalized.wav"
# with wave.open(wav_normalized_file_path, 'wb') as wf_norm:
#     wf_norm.setnchannels(CHANNELS)
#     wf_norm.setsampwidth(2)
#     wf_norm.setframerate(RATE)
#     wf_norm.writeframes(wav_normalized_data.tobytes())
# printt(f"放大後錄音存檔至 {wav_normalized_file_path}")

# at the end of this audio processing
beg_trans, end_trans, trans = online.finish()
if beg_trans is not None:
    print(f"{beg_trans:.2f} {end_trans:.2f} {trans} finish") # do something with current partial output
else:
    print("None finish")

online.init()  # refresh if you're going to re-use the object for the next audio