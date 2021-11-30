import tensorflow as tf
from tensorflow.python.framework.ops import Graph

from tf2_gnn import GNN, GNNInput

from model.GraphLayer import GraphLayer

from typing import List, Tuple

class GraphSequenceLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_state_size: int = 512) -> None:
        super(GraphSequenceLayer, self).__init__()
        
        self.gnn_params = GNN.get_default_hyperparameters()
        self.gnn_params['hidden_dim'] = 2

        self.annotations_layer = GNN(self.gnn_params)

        self.graph_level_layer = GraphLayer(hidden_state_size)

    def call(self, node_features: List[List[int]], adjacency_lists: Tuple[List[List[int]]], node_to_graph_map: List[int], batch_size: int, sequence_length: int):
        graph_outs = []

        annotations_input = tf.convert_to_tensor([[float(feature[0], feature[0])] for feature in node_features])
        adjacency_lists = tuple([tf.convert_to_tensor(adjacency_list, dtype=tf.int32) for adjacency_list in adjacency_lists if len(adjacency_list) != 0])

        for _ in range(sequence_length):
            layer_input = GNNInput(
                node_features = annotations_input,
                adjacency_lists = adjacency_lists,
                node_to_graph_map = tf.convert_to_tensor(node_to_graph_map),
                num_graphs = tf.convert_to_tensor(batch_size),
            )

            print('qweqwe', annotations_input.shape)

            annotations_level_out = self.annotations_layer(layer_input, training = True)
            graph_level_out = self.graph_level_layer(annotations_input, adjacency_lists, node_to_graph_map, batch_size)

            annotations_input = annotations_level_out

            graph_outs.append(graph_level_out)

        return tf.concat(graph_outs, axis=1)





