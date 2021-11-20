import numpy as np
import tensorflow as tf

class Generator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, batch_size=64,shuffle=True):
        'Initialization'
        self.data = data
        self.batch_size = batch_size
        
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data['src']) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        batch_src = tf.convert_to_tensor(np.array([self.data['src'][k] for k in indexes]))
        batch_tgt = tf.convert_to_tensor(np.array([self.data['tgt'][k] for k in indexes]))
        batch_trees = tf.convert_to_tensor(np.array([self.data['trees'][k] for k in indexes]))

        return np.array([batch_src,batch_tgt,batch_trees])

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data['src']))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    