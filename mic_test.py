import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import signal

from microphone import Microphone

from kapibara_audio import KapibaraAudio

import pyaudio
import numpy as np
from scipy.signal import butter,filtfilt,lfilter_zi,lfilter
import time

mic=Microphone(chunk=16000)

audio_device=pyaudio.PyAudio()

speaker=audio_device.open(16000,1,pyaudio.paInt16,output=True)

def design_butter_lowpass_filter(cutoff,fs,order):
    normal_cutoff = (2*cutoff) / fs
    print(normal_cutoff)
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='lowpass',analog=False)

    return b,a


b,a = design_butter_lowpass_filter(1200.0,16000.0,10)

print(b)
print(a)

iteration=0

try:
    while True:

        audio=mic.record(2)

        audio1=filtfilt(b,a,audio)

        audio1=audio1.astype(np.int16)

        plt.subplot(2,1,1)
        plt.plot(audio,label='raw')
        plt.subplot(2,1,2)
        plt.plot(audio1,label='filtered')
        plt.legend()
        plt.show()

        speaker.write(audio1,16000*2)

        #plt.plot(audio)

        #plt.show()
        print("iteration: ",iteration)
        time.sleep(2)

        #print("Press any key")
        #q=input()


        iteration=iteration+1
        #if q=='q':
        #    break
except KeyboardInterrupt:
    print("Finished")