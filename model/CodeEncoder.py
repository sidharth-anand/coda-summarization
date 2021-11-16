import tensorflow as tf
import numpy as np

from BinaryTreeLeafModule import BinaryTreeLeafModule
from BinaryTreeComposer import BinaryTreeComposer

from constants.constants import UNK

class CodeEncoder(tf.keras.Layer):
    def __init__(self, dictonaries, source_vocabulary_size: int, hidden_state_size:int, word_embedding_size: int, num_layers:int = 1) -> None:
        super(CodeEncoder, self).__init__()

        self.dictonaries = dictonaries
        self.source_vocabulary_size = source_vocabulary_size

        self.word_lut = tf.keras.layers.Embedding(source_vocabulary_size, word_embedding_size)
        self.leaf_module = BinaryTreeLeafModule(hidden_state_size)
        self.tree_composer = BinaryTreeComposer(hidden_state_size)

    def call(self, tree, lengths):
        if not tree.children:
            node = self.word_lut(tf.Tensor([self.dictonaries.lookup(tree.content, UNK)]))
            
            return self.leaf_module(node, lengths)
        elif tree.children:
            left_output, (left_current, left_hidden) = self.call(input, tree.children[0], lengths)
            right_output, (right_current, right_hidden) = self.call(input, tree.children[1], lengths)
            state = self.tree_composer(left_current, left_hidden, right_current, right_hidden)
            output = tf.concat([left_output, right_output], axis=-1)

            if not tree.parent:
                output = tf.expand_dims(output, axis=1)

                if np.max(lengths) > output.size()[0]:
                    output = tf.concat([output, tf.zeros((supl, output.size()[1], output.size()[2]))], axis=0)
                
                state[0] = tf.expand_dims(state[0], axis=1)
                state[1] = tf.expand_dims(state[1], axis=1)
            
            return output, state

                

