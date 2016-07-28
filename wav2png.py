#!/usr/bin/env python
from scipy.io import wavfile 
import sys
from pylab import *
import os

def show_wave_n_spec(speech):
    frame_rate, audio_array = wavfile.read(speech)
    subplot(211)
    ax1 = plot(audio_array[:32000])
    ylim(-0.5,0.5)
    title('Wave from and spectrogram of %s' % sys.argv[1])

    subplot(212)
    specgram(audio_array[:32000], NFFT=256, Fs=16000, noverlap=128)

    savefig(speech+'.png')
    clf()

fil = sys.argv[1]

if os.path.isfile(fil):
    show_wave_n_spec(fil)
elif os.path.isdir(fil):
    for f in os.listdir(fil):
        # print os.path.join(fil,f)
        if "wav" in f and "png" not in f:
            show_wave_n_spec(os.path.join(fil,f))

