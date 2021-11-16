import tensorflow as tf

from HybridAttention import HybridAttention
from StackedLSTM import StackedLSTM

class HybridDecoder(tf.keras.Layer):
    def __init__(self, num_layers: int, hidden_state_size: int, target_vocab_size: int, dropout: float = 0.3, use_attention: bool = True) -> None:
        super(HybridDecoder, self).__init__()

        self.hidden_state_size = hidden_state_size
        self.input_size = 2 *  hidden_state_size
        self.target_vocab_size = target_vocab_size
        self.dropout = dropout
        self.use_attention = use_attention

        self.word_lut = tf.keras.layers.Embedding(target_vocab_size, hidden_state_size)
        self.rnn = StackedLSTM(num_layers, self.input_size, dropout)
        self.dropout = tf.keras.layers.Dropout(dropout)

        if self.use_attention:
            self.attention = HybridAttention(hidden_state_size)
        else:
            self.attention = tf.keras.layers.Dense(self.inputs_size, use_bias=False)

    def step(self, embedding, output, tree_hidden, tree_context, text_hidden, text_context):
        embedding = tf.concat([embedding, output], axis=-1)

        tree_output, tree_hidden = self.rnn(embedding, tree_hidden)
        text_output, text_hidden = self.rnn(embedding, text_hidden)

        if self.use_attention:
            output, tree_attention, text_attention = self.attention(tree_output, tree_context, text_output, text_context)
        else:
            output = self.attention(tf.concat([tree_output, text_output], axis=-1))
        
        output = self.dropout(output)

        return output, tree_hidden, text_hidden

    def call(self, inputs, state):
        embedding, output, tree_hidden, tree_context, text_hidden, text_context = state

        embeddings = self.word_lut(inputs)
        outputs = []

        for i in range(inputs.shape[0]):
            output, tree_hidden, text_hidden = self.step(embedding, output, tree_hidden, tree_context, text_hidden, text_context)
            outputs.append(output)
            embedding = embeddings[i]

        outputs = tf.stack(outputs, axis=1)

        return outputs