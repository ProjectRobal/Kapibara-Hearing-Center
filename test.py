import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import copy

from tensorflow.keras import layers
from tensorflow.keras import models

BUFFER_SIZE = 16000*2

def load_and_prepare_audio(path):
    audio, _ = tf.audio.decode_wav(contents=tf.io.read_file(path))

    audio=tf.squeeze(audio, axis=-1)

    if audio.shape[0]<BUFFER_SIZE:
        zeros=tf.zeros(BUFFER_SIZE-audio.shape[0])
        audio=tf.concat([audio,zeros],0)

    if audio.shape[0]>BUFFER_SIZE:
        audio=tf.slice(audio,0,BUFFER_SIZE)

    audio=tf.cast(audio,dtype=tf.float32)

    spectrogram=tf.signal.stft(audio,frame_length=255,frame_step=128)

    spectrogram = tf.abs(spectrogram)

    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

spectrogram=load_and_prepare_audio('./samp194.wav')

model=tf.keras.models.load_model('./best_model')

prediction = model(spectrogram[None,...])

print(prediction)