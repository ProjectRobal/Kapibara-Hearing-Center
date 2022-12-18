import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import signal

from microphone import Microphone

from kapibara_audio import KapibaraAudio

import pyaudio
import numpy as np
from scipy.signal import butter,filtfilt
import time
import noisereduce as nr

'''
To do:

Filtr audio

 (cisza go denerwuje), usuń ciszę z przykładowych dźwięków

'''

mic=Microphone(chunk=16000)

model=KapibaraAudio('./best_model')

audio_device=pyaudio.PyAudio()

speaker=audio_device.open(16000,1,pyaudio.paInt16,output=True)

def design_butter_lowpass_filter(cutoff,fs,order):
    normal_cutoff = (2*cutoff) / fs
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='lowpass',analog=False)

    return b,a


b,a = design_butter_lowpass_filter(1200.0,16000.0,2)



iteration=0

try:
    while True:

        audio=mic.record(2)

        #plt.plot(audio)

        #plt.show()

        audio=filtfilt(b,a,audio).astype(np.int16)/32767.0

        #speaker.write((audio*32767).astype(np.int16),16000*2)

        #plt.plot(audio)

        #plt.show()

        #audio=nr.reduce_noise(y=audio,sr=16000)

        

        audio=tf.cast(audio,dtype=tf.float32)

        #plt.plot(audio)

        #plt.show()

        print("iteration: ",iteration)

        print(model.input(audio))

        time.sleep(2)

        #print("Press any key")
        #q=input()


        iteration=iteration+1
        #if q=='q':
        #    break
except KeyboardInterrupt:
    print("Finished")