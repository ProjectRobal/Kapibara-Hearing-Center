import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import signal

from microphone import Microphone

from kapibara_audio import KapibaraAudio

import pyaudio
import time

'''
To do:

Filtr audio

 (cisza go denerwuje), usuń ciszę z przykładowych dźwięków

'''

mic=Microphone(chunk=16000)

model=KapibaraAudio('./best_model')

print(model.input_wav('scary12.wav'))

iteration=0

try:
    while True:

        audio=mic.record(2)

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