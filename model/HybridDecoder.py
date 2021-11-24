import tensorflow as tf

from model.HybridAttention import HybridAttention
from model.StackedLSTM import StackedLSTM

class HybridDecoder(tf.keras.layers.Layer):
    def __init__(self, target_vocab_size: int, hidden_state_size: int = 512, num_layers: int = 1,   dropout: float = 0.3, use_attention: bool = True) -> None:
        super(HybridDecoder, self).__init__()

        self.hidden_state_size = hidden_state_size
        self.input_size = 2 *  hidden_state_size
        self.target_vocab_size = target_vocab_size
        self.dropout = dropout
        self.use_attention = use_attention

        self.word_lut = tf.keras.layers.Embedding(target_vocab_size, hidden_state_size)
        self.rnn = StackedLSTM(num_layers, hidden_state_size, dropout)
        self.dropout = tf.keras.layers.Dropout(dropout)

        if self.use_attention:
            self.attention = HybridAttention(hidden_state_size)
        else:
            self.attention = tf.keras.layers.Dense(self.inputs_size, use_bias=False)

    def step(self, embedding, output, tree_hidden: tuple, tree_context, text_hidden: tuple, text_context):
        print('asd')
        print('HD step() embedding shape ' , embedding.shape)
        print('HD step() output shape ' , output.shape)
        print('asd')

        embedding = tf.concat([embedding, output], axis=1)

        tree_output, tree_hidden = self.rnn(embedding, tree_hidden)
        text_output, text_hidden = self.rnn(embedding, text_hidden)

        print('HD step() tree o/p shape ' , tree_output.shape)
        print('HD step() text o/p shape ' , text_output.shape) 
        print('HD step() tree hidden shape ' , len(tree_hidden))
        print('HD step() text hidden shape ' , len(text_hidden)) 
        print('qwe')

        if self.use_attention:
            output, tree_attention, text_attention = self.attention(tree_output, tree_context, text_output, text_context)
        else:
            output = self.attention(tf.concat([tree_output, text_output], axis=-1))
        
        output = self.dropout(output)
        print('HD step() O/P shape ' ,  output.shape)

        return output, tree_hidden, text_hidden

    def call(self, inputs, state):
        embedding, output, tree_hidden, tree_context, text_hidden, text_context = state

        embeddings = self.word_lut(inputs)
        embeddings = tf.reshape(embeddings, (embeddings.shape[1], embeddings.shape[0], embeddings.shape[2]))
        
        outputs = []

        for i in range(inputs.shape[1]):
            output, tree_hidden, text_hidden = self.step(embedding, output, tree_hidden, tree_context, text_hidden, text_context)
            outputs.append(output)
            embedding = embeddings[i]

        outputs = tf.stack(outputs, axis=1)

        return outputs