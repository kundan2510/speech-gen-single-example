from scipy.io import wavfile
from scipy.signal import resample
import wave
import numpy as np

# wavio.py
# Author: Warren Weckesser
# License: BSD 3-Clause (http://opensource.org/licenses/BSD-3-Clause)
def _wav2array(nchannels, sampwidth, data):
    """data must be the string containing the bytes from the wav file."""
    num_samples, remainder = divmod(len(data), sampwidth * nchannels)
    if remainder > 0:
        raise ValueError('The length of data is not a multiple of '
                         'sampwidth * num_channels.')
    if sampwidth > 4:
        raise ValueError("sampwidth must not be greater than 4.")

    if sampwidth == 3:
        a = np.empty((num_samples, nchannels, 4), dtype=np.uint8)
        raw_bytes = np.fromstring(data, dtype=np.uint8)
        a[:, :, :sampwidth] = raw_bytes.reshape(-1, nchannels, sampwidth)
        a[:, :, sampwidth:] = (a[:, :, sampwidth - 1:sampwidth] >> 7) * 255
        result = a.view('<i4').reshape(a.shape[:-1])
    else:
        # 8 bit samples are stored as unsigned ints; others as signed ints.
        dt_char = 'u' if sampwidth == 1 else 'i'
        a = np.fromstring(data, dtype='<%s%d' % (dt_char, sampwidth))
        result = a.reshape(-1, nchannels)
    return result


def read_wav(file_path):
	audio_wav_file = wave.open(file_path)
	frame_rate = audio_wav_file.getframerate()
	num_channels = audio_wav_file.getnchannels()
	sample_width = audio_wav_file.getsampwidth()
	nframes = audio_wav_file.getnframes()
	data = audio_wav_file.readframes(nframes)
	audio_wav_file.close()
	audio_array = _wav2array(num_channels, sample_width, data)
	return frame_rate, sample_width, nframes, audio_array


def write_audio_file(name, frame_rate, data):
	data = data.astype('float32')
	data -= data.min()
	data /= data.max()
	data -= 0.5
	data *= 0.99999
	wavfile.write(name+'.wav', frame_rate, data)
