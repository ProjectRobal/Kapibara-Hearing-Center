'''
To do:

*Generate spectogram from audio data

*Load data into tensorflow dataset

*Create model

'''

#tensorflow imports
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import copy

from tensorflow.keras import layers
from tensorflow.keras import models

tf.config.run_functions_eagerly(True)

SAMPLE_RATE=16000

OUTPUT_SIZE=6

SAMPLE_TIME=2

BUFFER_SIZE=SAMPLE_RATE*SAMPLE_TIME

OUTPUT_SHAPE=0


def read_samples(dir,file="train.csv",delimiter=';'):
    
    audio=[]

    label=[]

    global OUTPUT_SHAPE

    with open(dir+"/"+file,"r") as f:
        headers=f.readline()

        OUTPUT_SHAPE=len(headers.split(delimiter))-1

        for line in f:
            objs=line.split(delimiter)

            for i in range(1,len(objs)):
                objs[i]=float(objs[i])

            audio.append(objs[0])

            label.append(tf.argmax(objs[1:]))

    return (audio,label)

def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(contents=audio_binary)

    return tf.squeeze(audio, axis=-1)

'''generate spectogram'''
def gen_spectogram(audio):
    
    audio=tf.cast(audio,dtype=tf.float32)

    spectrogram=tf.signal.stft(audio,frame_length=255,frame_step=128)

    spectrogram = tf.abs(spectrogram)

    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram


def load_audio(file,dir):
    audio= decode_audio(tf.io.read_file(dir+"/wavs/"+file+".wav"))
        
    if audio.shape[0]<BUFFER_SIZE:
        zeros=tf.zeros(BUFFER_SIZE-audio.shape[0])
        audio=tf.concat([audio,zeros],0)

    if audio.shape[0]>BUFFER_SIZE:
        audio=tf.slice(audio,0,BUFFER_SIZE)


def load_and_prepare_audio(files,dir):
    
    outputs=[]

    for entry in files:
        audio= decode_audio(tf.io.read_file(dir+"/wavs/"+entry+".wav"))
        
        if audio.shape[0]<BUFFER_SIZE:
            zeros=tf.zeros(BUFFER_SIZE-audio.shape[0])
            audio=tf.concat([audio,zeros],0)

        if audio.shape[0]>BUFFER_SIZE:
            audio=tf.slice(audio,0,BUFFER_SIZE)


        outputs.append(audio)
    
    return outputs



def gen_spectogram_list(audios):

    spectrograms=[]

    for audio in audios:
        spectrograms.append(gen_spectogram(audio))

    return spectrograms


files,labels=read_samples('./Dataset')

audios=load_and_prepare_audio(files,'./Dataset')


#print(data)
spectrograms=gen_spectogram_list(audios)

dataset=tf.data.Dataset.from_tensor_slices((spectrograms,labels))

print(dataset)
print(len(list(dataset)))

train_ds=dataset

batch_size=32

train_ds=train_ds.batch(batch_size)

train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)

valid_ds=train_ds.shuffle(batch_size,reshuffle_each_iteration=True)


for spectrogram, _ in dataset.take(1):
  input_shape = spectrogram.shape

print(input_shape)
print(OUTPUT_SHAPE)

num_labels=OUTPUT_SHAPE

# Instantiate the `tf.keras.layers.Normalization` layer.
norm_layer = layers.Normalization()
# Fit the state of the layer to the spectrograms
# with `Normalization.adapt`.
norm_layer.adapt(data=dataset.map(map_func=lambda spec, label: spec))

model = models.Sequential([
    layers.Input(shape=input_shape),
    # Downsample the input.
    layers.Resizing(64, 64),
    # Normalize.
    norm_layer,
    layers.Conv2D(64, 3, activation='relu'),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_labels),
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

EPOCHS = 100
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
)

metrics = history.history
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.show()

model.save("./best_model")