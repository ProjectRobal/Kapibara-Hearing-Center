import tensorflow as tf
import matplotlib.pyplot as plt

from microphone import Microphone

from kapibara_audio import KapibaraAudio


mic=Microphone(chunk=16000)

model=KapibaraAudio('./best_model')


try:
    while True:

        audio=mic.record(2)

        audio=tf.cast(audio,dtype=tf.float32)

        print(model.input(audio))

        print("Press any key")
        q=input()

        if q=='q':
            break
except KeyboardInterrupt:
    print("Finished")