import tensorflow as tf

from CodeEncoder import CodeEncoder
from TextEncoder import TextEncoder
from HybridDecoder import HybridDecoder
from Generator import Generator

#Rewrite this shit
class Hybrid2Seq(tf.keras.Model):
    def __init__(self, source_vocabulary_size: int, target_vocabulary_size: int) -> None:
        super(Hybrid2Seq, self).__init__()

        self.source_vocabulary_size = source_vocabulary_size
        self.target_vocabulary_size = target_vocabulary_size

        self.code_encoder = CodeEncoder(source_vocabulary_size)
        self.text_encoder = TextEncoder(target_vocabulary_size)
        self.hybrid_decoder = HybridDecoder(target_vocabulary_size)
        self.generator = Generator(target_vocabulary_size)
    
    def train_step(self, data):
        return super().train_step(data)