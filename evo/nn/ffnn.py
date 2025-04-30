from dataclasses import dataclass
from collections import defaultdict
from config import *
from genome import Genome
from nn.graphs import *
from functions import activations


@dataclass
class FeedForwardNetwork:
    inputs: list[int]
    outputs: list[int]
    edges: list[tuple[int, int, float]]  # source: int, target: int, weight: float
    nodes: dict[int, tuple[str, float]]  # act_func: str, bias: float
    layers: list[list[int]] = None  # Layers of nodes that can be computed in parallel
    incoming: dict[int, list[tuple[int, float]]] = None  # Adjacency list of incoming edges

    @staticmethod
    def from_genome(genome: Genome):
        # Convert genome to network
        edges = [(s, t, e.weight) for (s, t), e in genome.edges.items()]
        nodes = {i: (n.act_func, n.bias) for i, n in genome.nodes.items()}

        # Prune nodes that are not required for output
        required = required_for_output(genome.outputs, [(s,t) for s,t,_ in edges])
        nodes = {i: nodes[i] for i in required}

        # Prune edges
        edges = [(s, t, w) for s, t, w in edges if (s in nodes or s in genome.inputs) and t in nodes]

        return FeedForwardNetwork(genome.inputs, genome.outputs, edges, nodes)
    
    def __post_init__(self):
        # Collect layers of nodes that can be computed in parallel
        self.layers = feed_forward_layers(self.inputs, self.outputs, [(s,t) for s,t,_ in self.edges])

        # Create adjacency list to speed up search
        self.incoming = defaultdict(list)
        for s, t, w in self.edges:
            self.incoming[t].append((s, w))
        
    def forward(self, inputs: list[float]):
        # We can use memoization to speed up computation
        computed = defaultdict(float)

        # Initialize inputs
        for node, value in zip(self.inputs, inputs):
            computed[node] = value

        # Compute values of each node in each layer
        # We are helped by the fact that layers are independent sets
        for layer in self.layers:
            for node in layer:
                # Compute value of this node
                # f(x) = act_func(bias + sum(weight * f(source)))
                act_func, bias = self.nodes[node]
                computed[node] = activations[act_func](bias + sum(w * computed[s] for s, w in self.incoming[node]))
        
        # Return outputs
        return [computed[node] for node in self.outputs]

    def visualize(self, title=None):
        visualize(self.inputs, self.nodes, self.edges, title=title)
    
    def size(self):
        return len(self.nodes) + len(self.edges)


if __name__ == "__main__":
    genome = Genome.random(16, 15)

    network = FeedForwardNetwork.from_genome(genome)
    network.visualize()

    inputs = np.random.uniform(-1, 1, 16)
    print("Inputs:", inputs)
    print("Outputs:", network.forward(inputs))
