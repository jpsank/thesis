from dataclasses import dataclass
from nn.graphs import visualize
from config import *


@dataclass
class Node:
    id: int
    act_func: str = "tanh"
    bias: float = 0.0

    @staticmethod
    def random(id):
        # Create new random node
        return Node(id, bias=np.random.uniform(-1, 1))
    

@dataclass
class Edge:
    source: int
    target: int
    weight: float = 1.0

    @staticmethod
    def random(source, target):
        # Create new random edge between two nodes
        return Edge(source, target, np.random.uniform(-1, 1))


@dataclass
class Genome:
    inputs: list[int]
    outputs: list[int]
    nodes: dict[int, Node]
    edges: dict[tuple[int, int], Edge]

    mutate_add_node_prob: float = MUTATE_ADD_NODE
    mutate_add_edge_prob: float = MUTATE_ADD_EDGE
    mutate_remove_node_prob: float = MUTATE_REMOVE_NODE
    mutate_remove_edge_prob: float = MUTATE_REMOVE_EDGE

    @staticmethod
    def random(num_inputs, num_outputs):
        # Randomly initialize a fully connected network with no hidden neurons
        inputs = list(range(-num_inputs, 0))
        outputs = list(range(num_outputs))
        nodes = {i: Node.random(i) for i in range(num_outputs)}
        edges = {
            (i, j): Edge.random(i, j)
            for i in range(-num_inputs, 0)
            for j in range(num_outputs)
        }
        return Genome(inputs, outputs, nodes, edges)

    def copy(self):
        # Create a copy of the genome
        return Genome(
            self.inputs,
            self.outputs,
            {k: v for k, v in self.nodes.items()},
            {k: v for k, v in self.edges.items()},
            self.mutate_add_node_prob,
            self.mutate_add_edge_prob,
            self.mutate_remove_node_prob,
            self.mutate_remove_edge_prob
        )

    def size(self):
        # Return number of nodes and edges
        return len(self.nodes) + len(self.edges)

    def visualize(self, title=None):
        visualize(self.inputs, 
                  {k: (n.act_func, n.bias) for k, n in self.nodes.items()}, 
                  [(s, t, e.weight) for (s, t), e in self.edges.items()],
                  title=title)


def mutate_node(node: Node):
    # Mutate activation function
    if np.random.uniform() < MUTATE_ACT_FUNC:
        node.act_func = np.random.choice(ACTIVATIONS)
    
    # Mutate bias
    if np.random.uniform() < MUTATE_BIAS:
        node.bias += np.random.normal(scale=MUTATE_BIAS_SCALE)

def mutate_edge(edge: Edge):
    # Mutate weight
    if np.random.uniform() < MUTATE_WEIGHT:
        edge.weight += np.random.normal(scale=MUTATE_WEIGHT_SCALE)


def crossover_genomes(genome1: Genome, genome2: Genome):
    # Compare nodes
    nodes1 = set(genome1.nodes.keys())
    nodes2 = set(genome2.nodes.keys())
    homologous_nodes = nodes1 & nodes2
    disjoint_nodes = nodes1 ^ nodes2
    nodes = {}
    for node in homologous_nodes:
        nodes[node] = Node(
            node,
            genome1.nodes[node].act_func if np.random.uniform() < 0.5 else genome2.nodes[node].act_func,
            genome1.nodes[node].bias if np.random.uniform() < 0.5 else genome2.nodes[node].bias
        )
    for node in disjoint_nodes:
        nodes[node] = genome1.nodes[node] if node in nodes1 else genome2.nodes[node]

    # Compare edges
    edges1 = set(genome1.edges.keys())
    edges2 = set(genome2.edges.keys())
    homologous_edges = edges1 & edges2
    disjoint_edges = edges1 ^ edges2
    edges = {}
    for edge in homologous_edges:
        edges[edge] = Edge(
            edge[0], edge[1],
            genome1.edges[edge].weight if np.random.uniform() < 0.5 else genome2.edges[edge].weight
        )
    for edge in disjoint_edges:
        edges[edge] = genome1.edges[edge] if edge in edges1 else genome2.edges[edge]
    
    return Genome(genome1.inputs, genome1.outputs, nodes, edges)


def genomic_distance(genome1: Genome, genome2: Genome):
    # Compare nodes
    nodes1 = set(genome1.nodes.keys())
    nodes2 = set(genome2.nodes.keys())
    homologous_nodes = nodes1 & nodes2
    disjoint_nodes = nodes1 ^ nodes2
    node_distance = 0.0
    for node in homologous_nodes:
        node_distance += abs(genome1.nodes[node].bias - genome2.nodes[node].bias)
        node_distance += genome1.nodes[node].act_func != genome2.nodes[node].act_func
    node_distance += len(disjoint_nodes) * DISJOINT_COEFF

    # Compare edges
    edges1 = set(genome1.edges.keys())
    edges2 = set(genome2.edges.keys())
    homologous_edges = edges1 & edges2
    disjoint_edges = edges1 ^ edges2
    edge_distance = 0.0
    for edge in homologous_edges:
        edge_distance += abs(genome1.edges[edge].weight - genome2.edges[edge].weight)
    edge_distance += len(disjoint_edges) * DISJOINT_COEFF

    return node_distance / max(len(genome1.nodes), len(genome2.nodes)) + \
              edge_distance / max(len(genome1.edges), len(genome2.edges))

def mutate_genome(genome: Genome, next_node_id: int):
    # Mutate nodes
    for node in genome.nodes.values():
        mutate_node(node)
    
    # Mutate edges
    for edge in genome.edges.values():
        mutate_edge(edge)

    # Mutate genome
    if np.random.uniform() < genome.mutate_add_node_prob:
        if len(genome.edges) > 0:
            edges = list(genome.edges.keys())
            edge_to_split = edges[np.random.choice(len(edges))]
            source, target = edge_to_split
            genome.nodes[next_node_id] = Node.random(next_node_id)
            genome.edges[(source, next_node_id)] = Edge(source, next_node_id, 1.0)
            genome.edges[(next_node_id, target)] = Edge(next_node_id, target, genome.edges[edge_to_split].weight)
            del genome.edges[edge_to_split]
            next_node_id += 1
    
    if np.random.uniform() < genome.mutate_add_edge_prob:
        # Source cannot be an output node
        sources = list(set(genome.nodes.keys()) - set(genome.outputs))
        # Target cannot be an input node
        targets = list(set(genome.nodes.keys()) - set(genome.inputs))
        if len(sources) > 0 and len(targets) > 0:
            source = np.random.choice(sources)
            target = np.random.choice(targets)
            genome.edges[(source, target)] = Edge.random(source, target)
    
    if np.random.uniform() < genome.mutate_remove_node_prob:
        # Remove hidden node and all edges connected to it
        nodes = list(set(genome.nodes.keys()) - set(genome.inputs) - set(genome.outputs))
        if len(nodes) > 0:
            node_to_remove = np.random.choice(nodes)
            for edge in list(genome.edges.keys()):
                if edge[0] == node_to_remove or edge[1] == node_to_remove:
                    del genome.edges[edge]
            del genome.nodes[node_to_remove]
        
    if np.random.uniform() < genome.mutate_remove_edge_prob:
        # Remove random edge
        if len(genome.edges) > 0:
            edges = list(genome.edges.keys())
            del genome.edges[edges[np.random.choice(len(edges))]]
