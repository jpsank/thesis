from dataclasses import dataclass
from collections import defaultdict
from config import *
from genome import Genome
from nn.graphs import visualize, directed_from_input_to_output
from functions import activations


@dataclass
class RecurrentNetwork:
    inputs: list[int]
    outputs: list[int]
    edges: list[tuple[int, int, float]]  # source: int, target: int, weight: float
    nodes: dict[int, tuple[str, float]]  # act_func: str, bias: float

    incoming: dict[int, list[tuple[int, float]]] = None  # Adjacency list of incoming edges
    computed: defaultdict[int, float] = None # Computed output values of each node

    @staticmethod
    def from_genome(genome: Genome):
        # Convert genome to network
        edges = [(s, t, e.weight) for (s, t), e in genome.edges.items()]
        nodes = {i: (n.act_func, n.bias) for i, n in genome.nodes.items()}

        # Prune nodes that are not required for output
        required = directed_from_input_to_output(genome.inputs, genome.outputs, [(s,t) for s,t,_ in edges])
        nodes = {i: nodes[i] for i in required}

        # Prune edges
        edges = [(s, t, w) for s, t, w in edges if (s in nodes or s in genome.inputs) and t in nodes]

        return RecurrentNetwork(genome.inputs, genome.outputs, edges, nodes)
    
    def __post_init__(self):
        # Create adjacency list to speed up search
        self.incoming = defaultdict(list)
        for s, t, w in self.edges:
            self.incoming[t].append((s, w))
        
        # Initialize computed values
        self.computed = defaultdict(float)

    def forward(self, inputs: list[float]):
        # We can run depth-first search from each output node to compute its value
        # We can use memoization to speed up computation

        # Initialize inputs
        visited = set(self.inputs)
        for node, value in zip(self.inputs, inputs):
            self.computed[node] = value

        def dfs(node, upstream=set()):
            if node in visited or node in upstream:
                # If this node has already been computed, then use its computed value
                # If this node is already upstream and pending computation, then use its previous value
                # (prevents infinite loops in the case of cycles)
                return
            
            # Compute value of this node
            # f(x) = act_func(bias + sum(weight * f(source)))
            visited.add(node)
            act_func, bias = self.nodes[node]
            value = bias
            for source, weight in self.incoming[node]:
                dfs(source, upstream | {node})
                value += weight * self.computed[source]
            self.computed[node] = activations.get(act_func)(value)
        
        for node in self.outputs:
            dfs(node)
        
        return [self.computed[node] for node in self.outputs]

    def visualize(self, title=None):
        visualize(self.inputs, self.nodes, self.edges, title=title)
    
    def size(self):
        return len(self.nodes) + len(self.edges)


if __name__ == "__main__":
    genome = Genome.random(16, 15)

    network = RecurrentNetwork.from_genome(genome)
    network.visualize()

    inputs = np.random.uniform(-1, 1, 16)
    print("Inputs:", inputs)
    print("Outputs:", network.forward(inputs))
