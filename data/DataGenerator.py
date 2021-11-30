from typing import List

import numpy as np
import tensorflow as tf

from data.Graph import merge_graphs

from constants.constants import PAD


class DataGenerator(tf.compat.v1.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, data, target_vocabulary_size: int, batch_size=64, shuffle=True):
        'Initialization'

        self.data = data
        self.batch_size = batch_size
        self.target_vocabulary_size = target_vocabulary_size

        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data['src']) / self.batch_size))

    def __batchify__(self, data: List[List[int]], include_lengths=False):
        lengths = [len(x) for x in data]
        max_length = np.amax(lengths)

        for sentence in data:
            length = len(sentence)
            for _ in range(max_length - length):
                sentence.append(PAD)

        if include_lengths:
            return data, lengths
        else:
            return data

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index *
                               self.batch_size: (index + 1) * self.batch_size]

        # Find list of IDs
        batch_src = [self.data['src'][k] for k in indexes]
        batch_tgt = [self.data['tgt'][k] for k in indexes]
        batch_trees = [self.data['trees'][k] for k in indexes]
        batch_leafs = [[0] * 64] * 64

        batch_src, src_lengths = self.__batchify__(
            batch_src, include_lengths=True)
        batch_leafs, leaf_lengths = self.__batchify__(
            batch_leafs, include_lengths=True)
        batch_tgt = self.__batchify__(batch_tgt)

        tree_lengths = []
        node_to_graph_map = list(np.concatenate([[i] * len(tree['node_list']) for i, tree in enumerate(batch_trees)]).flat)

        merged_tree_batch = batch_trees[0]
        for tree in batch_trees[1:]:
            merged_tree_batch = merge_graphs(merged_tree_batch, tree)

        print('nodes', merged_tree_batch['node_list'])

        return (
            (
                (self.make_tensor(batch_src), src_lengths),
                (merged_tree_batch, tree_lengths, (self.make_tensor(batch_leafs), leaf_lengths), node_to_graph_map),
                tf.convert_to_tensor(batch_tgt),
                range(len(batch_src)),
                tf.one_hot(np.asarray(batch_tgt), self.target_vocabulary_size)
            ),
        )

    def make_tensor(self, data):
        return tf.convert_to_tensor(data)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data['src']))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
