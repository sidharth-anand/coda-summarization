import tensorflow as tf

class BinaryTreeComposer(tf.keras.layers.Layer):
    def __init__(self, hidden_state_size: int, gate_output: bool=False) -> None:
        super(BinaryTreeComposer, self).__init__()

        self.hidden_state_size = hidden_state_size
        self.gate_output = gate_output

        self.ilh, self.irh = self.create_gate(hidden_state_size, '1')
        self.lflh, self.lfrh = self.create_gate(hidden_state_size, '2')
        self.rflh, self.rfrh = self.create_gate(hidden_state_size, '3')
        self.ulh, self.urh = self.create_gate(hidden_state_size, '4')


    def create_gate(self, size: int, name: str) -> tuple:
        lh = tf.keras.layers.Dense(size, use_bias=True, name=f'{name}_lh')
        rh = tf.keras.layers.Dense(size, use_bias=True, name=f'{name}_rh')
        return lh, rh

    def call(self, lc: tf.Tensor, lh: tf.Tensor, rc: tf.Tensor, rh: tf.Tensor) -> tuple:
        i = tf.keras.activations.sigmoid(tf.math.add(self.ilh(lh), self.irh(rh)))
        lf = tf.keras.activations.sigmoid(tf.math.add(self.lflh(lh), self.lfrh(rh)))
        rf = tf.keras.activations.sigmoid(tf.math.add(self.rflh(lh), self.rfrh(rh)))
        update = tf.math.tanh(tf.math.add(self.ulh(lh), self.urh(rh)))
        
        #Tensor of (1, hidden_state_size)
        current = tf.math.add(tf.math.multiply(i, update), tf.math.multiply(lf, lc), tf.math.multiply(rf, rc))
        

        
        
        #Tensor of (1, hidden_state_size)
        hidden = tf.math.tanh(current)
        
        return current, hidden
