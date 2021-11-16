import tensorflow as tf

class HybridAttention(tf.keras.Layer):
    def __init__(self, hidden_state_size: int) -> None:
        super(HybridAttention, self).__init__()

        self.hidden_state_size = hidden_state_size
        
        self.linear_in = tf.keras.layers.Dense(hidden_state_size, use_bias=False)
        self.softmax = tf.keras.layers.Softmax()
        self.linear_out = tf.keras.layers.Dense(hidden_state_size * 4, use_bias=False)
        self.tanh = tf.keras.layers.Activation('tanh')

        self.tree_mask = None
        self.text_mask = None

    def apply_mask(self, tree_mask, text_mask):
        self.tree_mask = tree_mask
        self.text_mask = text_mask

    def call(self, tree_inputs, tree_context, text_inputs, text_context):
        tree_target = tf.expand_dims(self.linear_in(tree_inputs), axis=2)
        text_target = tf.expand_dims(self.linear_in(text_inputs), axis=2)

        tree_attention = tf.squeeze(tf.matmul(tree_context, tree_target), axis=2)
        text_attention = tf.squeeze(tf.matmul(text_context, text_target), axis=2)

        if self.tree_mask is not None and self.text_mask is not None:
            tree_attention = tf.math.multiply(tree_attention, self.tree_mask)
            tree_attention = self.softmax(tree_attention)

            text_attention = tf.math.multiply(text_attention, self.text_mask)
            text_attention = self.softmax(text_attention)

        tree_attention_3d = tf.reshape(tree_attention.shape[0], 1, tree_attention.shape[1])
        text_attention_3d = tf.reshape(text_attention.shape[0], 1, text_attention.shape[1])

        tree_weighted_context = tf.squeeze(tf.matmul(tree_attention_3d, tree_context), axis=0)
        tree_combined_context = tf.concat([tree_weighted_context, tree_inputs], axis=1)

        text_weighted_context = tf.squeeze(tf.matmul(text_attention_3d, text_context), axis=0)
        text_combined_context = tf.concat([text_weighted_context, text_inputs], axis=1)

        combined_context = tf.concat([tree_combined_context, text_combined_context], axis=1)
        combined_context = self.tanh(self.linear_out(combined_context))

        return combined_context, tree_attention, text_attention