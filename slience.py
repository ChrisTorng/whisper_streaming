# https://github.com/spatialaudio/python-sounddevice/issues/44

import sounddevice as sd
import numpy as np
from scipy.io import wavfile
import time

RATE = 16000 # 8000, 16000, 32000

rec = sd.rec(RATE * 2, samplerate=RATE, channels=1)
print('  sd.wait() ->', sd.wait())
for channel in range(rec.shape[1]):
    non_silence = np.where(rec[:,channel] > float(0))[0]
    print('  Ch. {}, initial silence {} samples'.format(channel, non_silence[0]))
wavfile.write(f'audio/temp/RECORDED-{str(time.time())}.wav', RATE, rec)