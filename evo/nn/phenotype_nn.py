from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from ..nn.ffnn import FeedForwardNetwork
from ..config import *
from ..functions import activations


@dataclass
class PhenotypeNetwork:
    substrate: list[list[int]]
    weights: list[np.ndarray]
    biases: list[np.ndarray]
    act_func: str

    @staticmethod
    def create(cppn: FeedForwardNetwork, substrate: list[list[int]], act_func: str):
        # Create phenotype network from CPPN applied to substrate
        n_neurons = [np.prod(dim) for dim in substrate]
        weights = []
        biases = []
        for l in range(1, len(substrate)):
            W = np.zeros((n_neurons[l], n_neurons[l-1]))
            b = np.zeros(n_neurons[l])
            for j, index_j in enumerate(np.ndindex(*substrate[l])):
                # Normalize index to [-1, 1]
                index_j = [0 if substrate[l][d] == 1 else index_j[d] / (substrate[l][d] - 1) * 2 - 1 for d in range(len(substrate[l])) ]
                # print(index_j)
                for i, index_i in enumerate(np.ndindex(*substrate[l-1])):
                    index_i = [0 if substrate[l-1][d] == 1 else index_i[d] / (substrate[l-1][d] - 1) * 2 - 1 for d in range(len(substrate[l-1])) ]
                    W[j, i] = cppn.forward([*index_j, *index_i])[0]
                b[j] = cppn.forward([*index_j, -1])[0]
            # print()

            weights.append(W)
            biases.append(b)
        
        return PhenotypeNetwork(substrate, weights, biases, act_func)
    
    def forward(self, inputs: np.ndarray):
        # Forward propagate inputs through network
        assert inputs.shape == self.substrate[0]
        # print("Inputs:", np.dot(inputs, [0.2989, 0.5870, 0.1140]))
        curr = inputs.flatten()
        for W, b in zip(self.weights, self.biases):
            curr = activations[self.act_func](np.dot(W, curr) + b)
            # print(curr)
        # print()
        return curr.reshape(self.substrate[-1])

    def size(self):
        # Return number of weights and biases
        return sum([w.size for w in self.weights]) + sum([b.size for b in self.biases])


@dataclass
class RecurrentPhenotypeNetwork(PhenotypeNetwork):
    hidden_weights: list[np.ndarray]  # Recurrent connections
    hidden_state: list[np.ndarray]  # Hidden state

    @staticmethod
    def create(cppn: FeedForwardNetwork, substrate: list[list[int]], act_func: str):
        # Create phenotype network from CPPN applied to substrate
        n_neurons = [np.prod(dim) for dim in substrate]
        weights = []
        biases = []
        hidden_weights = []
        for l in range(1, len(substrate)):
            W = np.zeros((n_neurons[l], n_neurons[l-1]))
            h = np.zeros((n_neurons[l], n_neurons[l-1]))
            b = np.zeros(n_neurons[l])
            for j, index_j in enumerate(np.ndindex(*substrate[l])):
                # Normalize index to [-1, 1]
                index_j = [0 if substrate[l][d] == 1 else index_j[d] / (substrate[l][d] - 1) * 2 - 1 for d in range(len(substrate[l])) ]
                for i, index_i in enumerate(np.ndindex(*substrate[l-1])):
                    index_i = [0 if substrate[l-1][d] == 1 else index_i[d] / (substrate[l-1][d] - 1) * 2 - 1 for d in range(len(substrate[l-1])) ]
                    W[j, i] = cppn.forward([*index_j, *index_i])[0]
                    h[j, i] = cppn.forward([*index_i, *index_j])[0]
                b[j] = cppn.forward([*index_j, -1])[0]

            weights.append(W)
            biases.append(b)
            hidden_weights.append(h)
        
        # Initialize hidden state
        empty_state = [np.zeros(n) for n in n_neurons]

        return RecurrentPhenotypeNetwork(substrate, weights, biases, act_func, hidden_weights, empty_state)
    
    def forward(self, inputs: np.ndarray):
        # Forward propagate inputs through network
        assert inputs.shape == self.substrate[0]
        curr = [inputs.flatten()]
        for W, b, h, prev in zip(self.weights, self.biases, self.hidden_weights, self.hidden_state):
            curr.append(activations[self.act_func](np.dot(W, curr[-1]) + np.dot(h, prev) + b))
        self.hidden_state = curr
        return curr[-1].reshape(self.substrate[-1])
    
    def size(self):
        return super().size() + sum([h.size for h in self.hidden_weights])
