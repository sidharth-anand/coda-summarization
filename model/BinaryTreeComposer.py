import tensorflow as tf

class BinaryTreeComposer(tf.keras.Layer):
    def __init__(self, hidden_state_size: int, gate_output: bool=False) -> None:
        super(BinaryTreeComposer, self).__init__()

        self.hidden_state_size = hidden_state_size
        self.gate_output = gate_output

        self.ilh, self.irh = self.create_gate(hidden_state_size)
        self.lflh, self.lfrh = self.create_gate(hidden_state_size)
        self.rflh, self.rfrh = self.create_gate(hidden_state_size)
        self.ulh, self.urh = self.create_gate(hidden_state_size)

        if gate_output:
            self.olh, self.orh = self.create_gate(hidden_state_size)

    def create_gate(self, size: int):
        lh = tf.keras.layers.Dense(size, use_bias=True)
        rh = tf.keras.layers.Dense(size, use_bias=True)
        return lh, rh

    def call(self, lc, lh, rc, rh):
        i = tf.keras.activations.sigmoid(tf.math.add(self.ilh(lh), self.irh(rh)))
        lf = tf.keras.activations.sigmoid(tf.math.add(self.lflh(lh), self.lfrh(rh)))
        rf = tf.keras.activations.sigmoid(tf.math.add(self.rflh(lh), self.rfrh(rh)))
        update = tf.math.tanh(tf.math.add(self.ulh(lh), self.urh(rh)))
        
        current = tf.math.add(tf.math.multiply(i, update), tf.math.multiply(lf, lc), tf.math.multiply(rf, rc))

        if self.gate_output:
            output = tf.keras.activations.sigmoid(tf.math.add(self.olh(lh), self.orh(rh)))
            hidden = tf.math.multiply(output, tf.math.tanh(current))
        else:
            hidden = tf.math.tanh(current)
        
        return current, hidden
