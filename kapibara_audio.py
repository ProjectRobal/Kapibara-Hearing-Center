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

        neutral=[]

        unsettling=[]

        pleasent=[]

        scary=[]

        nervourses=[]

        with open(dir+"/"+file,"r") as f:
            headers=f.readline()
            for line in f:
                objs=line.split(delimiter)

                for i in range(1,len(objs)):
                    objs[i]=objs[i].replace(',','.')
                    objs[i]=float(objs[i])

                audio.append(objs[0])

                neutral.append(objs[1])

                unsettling.append(objs[2])

                pleasent.append(objs[3])

                scary.append(objs[4])

                nervourses.append(objs[5])

        
        return (audio,(neutral,unsettling,pleasent,scary,nervourses))




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

        train_ds = train_ds.take(1).cache().prefetch(tf.data.AUTOTUNE)

        valid_ds=train_ds.shuffle(batch_size,reshuffle_each_iteration=True)

        for spectrogram, _ in dataset.take(1):
            input_shape = spectrogram.shape

        #a root 
        input_layer=layers.Input(shape=input_shape)

        resizing=layers.Resizing(64,64)(input_layer)

        # Instantiate the `tf.keras.layers.Normalization` layer.
        norm_layer = layers.Normalization()
        # Fit the state of the layer to the spectrograms
        # with `Normalization.adapt`.
        norm_layer.adapt(data=dataset.map(map_func=lambda spec, label: spec))

        conv1=layers.Conv2D(64, 3, activation='relu')(resizing)

        conv2=layers.Conv2D(128, 3, activation='relu')(conv1)

        maxpool=layers.MaxPooling2D()(conv2)

        dropout1=layers.Dropout(0.25)(maxpool)

        root_output=layers.Flatten()(dropout1)

        #output layers

        neutral=layers.Dense(128, activation='relu')(root_output)

        neutral1=layers.Dense(64, activation='relu')(neutral)

        neutral_output=layers.Dense(1,activation='relu',name='neutral')(neutral1)

        unsettling=layers.Dense(128, activation='relu')(root_output)

        unsettling1=layers.Dense(64, activation='relu')(unsettling)

        unsettling_output=layers.Dense(1,activation='relu',name='unsettling')(unsettling1)

        pleasent=layers.Dense(128, activation='relu')(root_output)

        pleasent1=layers.Dense(64, activation='relu')(pleasent)

        pleasent_output=layers.Dense(1,activation='relu',name='pleasent')(pleasent1)

        scary=layers.Dense(128, activation='relu')(root_output)

        scary1=layers.Dense(64, activation='relu')(scary)

        scary_output=layers.Dense(1,activation='relu',name='scary')(scary1)

        nervourses=layers.Dense(128, activation='relu')(root_output)

        nervourses1=layers.Dense(64, activation='relu')(nervourses)

        nervourses_output=layers.Dense(1,activation='relu',name='nervourses')(nervourses1)

        model=models.Model(inputs=input_layer,outputs=[neutral_output,unsettling_output,pleasent_output,scary_output,nervourses_output])

        model.summary()

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss={'neutral':'mse', 
            'unsettling':'mse',
            'pleasent':'mse',
            'scary':'mse',
            'nervourses':'mse'},
            metrics={'neutral':tf.keras.metrics.RootMeanSquaredError(), 
            'unsettling':tf.keras.metrics.RootMeanSquaredError(),
            'pleasent':tf.keras.metrics.RootMeanSquaredError(),
            'scary':tf.keras.metrics.RootMeanSquaredError(),
            'nervourses':tf.keras.metrics.RootMeanSquaredError()},
        )

        
        history = model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=EPOCHS
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
        print(prediction)

        mean=0

        for p in prediction:
            p=p.numpy()[0]
            mean=mean+p

        output=[]

        for p in prediction:
            p=p.numpy()[0]
            output.append(p/mean)

        return output

    '''audio - raw audio input'''
    def input(self,audio):

        if audio.shape[0]<BUFFER_SIZE:
            zeros=tf.zeros(BUFFER_SIZE-audio.shape[0])
            audio=tf.concat([audio,zeros],0)

        if audio.shape[0]>BUFFER_SIZE:
            audio=tf.slice(audio,0,BUFFER_SIZE)

        spectrogram=self.gen_spectogram(audio)[None,..., tf.newaxis]

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
        


