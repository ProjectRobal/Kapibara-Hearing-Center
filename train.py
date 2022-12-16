import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


tf.config.run_functions_eagerly(True)

from kapibara_audio import KapibaraAudio



model=KapibaraAudio()


history = model.train("./Dataset",EPOCHS=50,batch_size=64)

metrics = history.history

plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.show()

