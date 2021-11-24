import tensorflow as tf

class HybridAttention(tf.keras.layers.Layer):
    def __init__(self, hidden_state_size: int) -> None:
        super(HybridAttention, self).__init__()

        self.hidden_state_size = hidden_state_size
        
        self.linear_in = tf.keras.layers.Dense(hidden_state_size, use_bias=False)
        self.softmax = tf.keras.layers.Softmax()
        self.linear_out = tf.keras.layers.Dense(hidden_state_size, use_bias=False)
        self.tanh = tf.keras.layers.Activation('tanh')

        self.tree_mask = None
        self.text_mask = None

    def apply_mask(self, tree_mask, text_mask) -> None:
        self.tree_mask = tree_mask
        self.text_mask = text_mask

    def call(self, tree_inputs: tf.Tensor, tree_context: tf.Tensor, text_inputs: tf.Tensor, text_context: tf.Tensor) -> tuple:
        print('hybrid attention')

        print("Tree input shape - 1 " + tree_inputs.shape)
        print("Text input shape - 1 " + text_inputs.shape)

        tree_context = tf.reshape(tree_context, [tree_context.shape[1], tree_context.shape[0], tree_context.shape[2]])
        text_context = tf.reshape(text_context, [text_context.shape[1], text_context.shape[0], text_context.shape[2]])

        tree_target = tf.expand_dims(self.linear_in(tree_inputs), axis=2)
        text_target = tf.expand_dims(self.linear_in(text_inputs), axis=2)

        print("Tree input shape - 2 " + tree_inputs.shape)
        print("Text input shape - 2 " + text_inputs.shape)

        tree_attention = tf.squeeze(tf.matmul(tree_context, tree_target), axis=2)
        text_attention = tf.squeeze(tf.matmul(text_context, text_target), axis=2)

        print("Tree attention shape " + tree_attention.shape)
        print("Text attention shape " + text_attention.shape)

        if self.tree_mask is not None and self.text_mask is not None:
            tree_attention = tf.math.multiply(tree_attention, self.tree_mask)
            tree_attention = self.softmax(tree_attention)

            text_attention = tf.math.multiply(text_attention, self.text_mask)
            text_attention = self.softmax(text_attention)

        tree_attention_3d = tf.reshape(tree_attention, (tree_attention.shape[0], 1, tree_attention.shape[1]))
        text_attention_3d = tf.reshape(text_attention, (text_attention.shape[0], 1, text_attention.shape[1]))

        print("Tree attention 3D shape " +tree_attention_3d.shape)
        print("Text attention 3D shape " +text_attention_3d.shape)

        tree_weighted_context = tf.squeeze(tf.matmul(tree_attention_3d, tree_context), axis=1)
        tree_combined_context = tf.concat([tree_weighted_context, tree_inputs], axis=1)

        text_weighted_context = tf.squeeze(tf.matmul(text_attention_3d, text_context), axis=1)
        text_combined_context = tf.concat([text_weighted_context, text_inputs], axis=1)

        combined_context = tf.concat([tree_combined_context, text_combined_context], axis=1)
        combined_context = self.tanh(self.linear_out(combined_context))

        print("Tree weighted context shape " +tree_weighted_context.shape)
        print("Text weighted context shape " +text_weighted_context.shape)
        print("Tree weighted context shape " +combined_context.shape)

        return combined_context, tree_attention, text_attention