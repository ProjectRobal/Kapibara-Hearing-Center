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

        with open("samp.wav","wb") as f:
            out=tf.audio.encode_wav(tf.reshape(audio/2**16-1,[model.buffer_size,1]),16000)
            f.write(out.numpy())

        plt.plot(audio)

        plt.show()

        print(model.input(audio))

        print("Press any key")
        q=input()

        if q=='q':
            break
except KeyboardInterrupt:
    print("Finished")