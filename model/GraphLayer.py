import tensorflow as tf
from tf2_gnn import GNN, GNNInput
from tf2_gnn.layers import NodesToGraphRepresentationInput, WASGraphRepresentation
from typing import List, Tuple

from tf2_gnn.layers.nodes_to_graph_representation import NodesToGraphRepresentation

class GraphLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_state_size: int = 512) -> None:
        super(GraphLayer, self).__init__()
        self.gnn_params = GNN.get_default_hyperparameters()
        # self.these_hypers = {
        #     "message_calculation_class": "rgcn",
        #     "initial_node_representation_activation": "tanh",
        #     "dense_intermediate_layer_activation": "tanh",
        #     "num_layers": 4,
        #     "dense_every_num_layers": 2,
        #     "residual_every_num_layers": 2,
        #     "use_inter_layer_layernorm": False,
        #     "hidden_dim": 16,
        #     "layer_input_dropout_rate": 0.0,
        #     "global_exchange_mode": "gru",  # One of "mean", "mlp", "gru"
        #     "global_exchange_every_num_layers": 2,
        #     "global_exchange_weighting_fun": "softmax",  # One of "softmax", "sigmoid"
        #     "global_exchange_num_heads": 4,
        #     "global_exchange_dropout_rate": 0.2,
        # } 
        self.gnn_input = GNN(self.gnn_params)
        self.nodes_to_graph_input = WASGraphRepresentation(
            graph_representation_size = hidden_state_size
        )
        
    def call(self, node_features: tf.Tensor, adjacency_lists: Tuple[tf.Tensor, ...], node_to_graph_map: List[int], batch_size: int):
        layer_input = GNNInput(
            node_features = node_features,
            adjacency_lists = adjacency_lists,
            node_to_graph_map = tf.convert_to_tensor(node_to_graph_map),
            num_graphs = tf.convert_to_tensor(batch_size),
        )
        layer_output = self.gnn_input(layer_input)

        nodes_input = NodesToGraphRepresentationInput(
            node_embeddings = layer_output,
            node_to_graph_map = tf.convert_to_tensor(node_to_graph_map),
            num_graphs = tf.convert_to_tensor(batch_size)
        )
        nodes_output = self.nodes_to_graph_input(nodes_input)



        return nodes_output
