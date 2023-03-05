import numpy as np
from scipy.io import wavfile


def read_wav(name):  # output: float32 or float64, [-1.0, (1.0)]
    rate, data = wavfile.read(name)
    dtype = data.dtype
    if dtype.kind != 'f':
        data = data.astype(np.promote_types(dtype, np.float32))
        scale = np.iinfo(dtype).max + 1
        if dtype.kind == 'u':
            scale /= 2
            data -= scale
        data /= scale
    return rate, data, dtype


def write_wav(name, rate, data, outdtype):  # input: float, [-1.0, (1.0)]
    if data.dtype != outdtype:
        if outdtype.kind == 'f':
            data = data.astype(outdtype)
        else:
            middtype = np.promote_types(data.dtype, outdtype)
            if data.dtype != middtype:
                data = data.astype(middtype)
            scale = np.iinfo(outdtype).max + 1
            if outdtype.kind == 'u':
                scale /= 2
                data += 1.0
            data = np.rint(data * scale).clip(np.iinfo(outdtype).min, np.iinfo(outdtype).max).astype(outdtype)
    wavfile.write(name, rate, data)
