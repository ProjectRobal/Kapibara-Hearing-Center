import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models

BUFFER_SIZE = 16000*2


class KapibaraAudio:
    '''path - a path to a model'''
    def __init__(self,path=None):
        self.model=None
        if path is not None:
            self.model=tf.keras.models.load_model(path)
        self.answers=['neutral','disturbing','unpleasent','pleasent','scary','irritanting']
        self.sample_rate=16000
        self.buffer_size=BUFFER_SIZE

    '''read samples from dataset'''
    def read_samples(self,dir,file="train.csv",delimiter=';'):
    
        audio=[]

        label=[]

        with open(dir+"/"+file,"r") as f:
            headers=f.readline()
            for line in f:
                objs=line.split(delimiter)

                for i in range(1,len(objs)):
                    objs[i]=float(objs[i])

                audio.append(objs[0])

                label.append(tf.argmax(objs[1:]))
        
        return (audio,label)


    '''path - a path to dataset'''
    def train(self,path,batch_size=32,EPOCHS = 100,file="train.csv",delimiter=";",save_path="./best_model"):
        
        files,labels = self.read_samples(path,file,delimiter)

        spectrograms=[]

        for file in files:
            audio=self.load_wav(path+"/wavs/"+file+".wav")

            spectrograms.append(self.gen_spectogram(audio))

        print("Samples count: ",len(spectrograms))

        dataset=tf.data.Dataset.from_tensor_slices((spectrograms,labels))

        train_ds=dataset

        train_ds=train_ds.batch(batch_size)

        train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)

        valid_ds=train_ds.shuffle(batch_size,reshuffle_each_iteration=True)

        for spectrogram, _ in dataset.take(1):
            input_shape = spectrogram.shape

        num_labels=len(self.answers)

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

        
        history = model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=EPOCHS,
            callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
            )

        model.save(save_path)

        return history


    '''generate spectogram'''
    def gen_spectogram(self,audio):
    
        spectrogram=tf.signal.stft(audio,frame_length=255,frame_step=128)

        spectrogram = tf.abs(spectrogram)

        spectrogram = spectrogram[..., tf.newaxis]
        return spectrogram

    def get_result(self,prediction):

        prediction=prediction.numpy()[0]
        print
        max_v=max(prediction)

        for i in range(len(prediction)):
            if prediction[i]==max_v:
                return self.answers[i]
        
        return None

    '''audio - raw audio input'''
    def input(self,audio):

        if audio.shape[0]<BUFFER_SIZE:
            zeros=tf.zeros(BUFFER_SIZE-audio.shape[0])
            audio=tf.concat([audio,zeros],0)

        if audio.shape[0]>BUFFER_SIZE:
            audio=tf.slice(audio,0,BUFFER_SIZE)

        spectrogram=self.gen_spectogram(audio)

        prediction = self.model(spectrogram)

        return self.get_result(prediction)

    '''path - a path to the wav file'''
    def load_wav(self,path):
        audio, _ = tf.audio.decode_wav(contents=tf.io.read_file(path))

        audio=tf.squeeze(audio, axis=-1)

        if audio.shape[0]<BUFFER_SIZE:
            zeros=tf.zeros(BUFFER_SIZE-audio.shape[0])
            audio=tf.concat([audio,zeros],0)

        if audio.shape[0]>BUFFER_SIZE:
            audio=tf.slice(audio,0,BUFFER_SIZE)

        audio=tf.cast(audio,dtype=tf.float32)

        return audio

    '''path - a path to the wav file'''
    def input_wav(self, path):

        audio=self.load_wav(path)

        return self.input(audio)
        


