import tensorflow as tf
import numpy as np
from data.Tree import Tree

from model.BinaryTreeLeafModule import BinaryTreeLeafModule
from model.BinaryTreeComposer import BinaryTreeComposer

from constants.constants import UNK
from data.Dictionary import Dictionary

class CodeEncoder(tf.keras.layers.Layer):
    def __init__(self, dictionaries: Dictionary, source_vocabulary_size: int, hidden_state_size: int = 512, word_embedding_size: int = 512) -> None:
        super(CodeEncoder, self).__init__()

        self.dictonaries = dictionaries
        self.source_vocabulary_size = source_vocabulary_size

        self.word_lut = tf.keras.layers.Embedding(source_vocabulary_size, word_embedding_size)
        self.leaf_module = BinaryTreeLeafModule(hidden_state_size)
        self.tree_composer = BinaryTreeComposer(hidden_state_size)

    def call(self, tree: Tree, lengths: int) -> tuple:
        if not tree.children:
            node = self.word_lut(tf.convert_to_tensor([self.dictonaries.lookup(tree.content, UNK)]))
            return self.leaf_module(node)
        elif tree.children:
            left_output, (left_current, left_hidden) = self.call(tree.children[0], lengths)
            right_output, (right_current, right_hidden) = self.call(tree.children[1], lengths)
           
            state = self.tree_composer(left_current, left_hidden, right_current, right_hidden)
            output = tf.concat([left_output, right_output], axis=0)

            #Output - Tensor of shape (1, batch_size, word_embeddings)
            if not tree.parent:
                output = tf.expand_dims(output, axis=0)

                if np.max(lengths) > output.shape[1]:
                    output = tf.concat([output, tf.zeros((output.shape[0], np.max(lengths) - output.shape[1], output.shape[2]))], axis=1)
                
                #Tuple of length 2
                state = (tf.expand_dims(state[0], axis=1), tf.expand_dims(state[1], axis=1))
            
            return output, state

                

