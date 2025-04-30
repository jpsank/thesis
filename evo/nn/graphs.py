import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def visualize(inputs: list[int], nodes: dict[int, tuple[str, float]], edges: list[tuple[int, int, float]], title=None):
    # Create directed graph
    G = nx.DiGraph()
    G.add_nodes_from(inputs)
    G.add_nodes_from(nodes.keys())
    G.add_weighted_edges_from(edges)

    labels = {i: f"{i}:\n{act_func}\n{bias:.2f}" for i, (act_func, bias) in nodes.items()}
    labels.update({i: f"{i}:\ninput" for i in inputs})

    fig = plt.figure(figsize=(10,5))
    if title:
        fig.suptitle(title)
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    nx.draw(G, pos=pos, labels=labels, with_labels=True, node_color="lightblue", node_size=500, font_size=6, font_weight="bold")
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels={(s, t): f"{w:.2f}" for s, t, w in edges}, font_size=6)
    plt.show()


def required_for_output(outputs: list[int], edges: list[tuple[int, int]]):
    # Want to find: all hidden nodes that lie on a path to any output node
    # (assumes input nodes have negative ids)
    
    # Create adjacency list to speed up search
    incoming = defaultdict(set)
    for s, t in edges:  # O(|E|)
        incoming[t].add(s)
    
    # We can run breadth-first search from each output node
    required = set()

    def bfs(node):  # O(|V| + |E|)
        if node < 0:
            # Input node
            return
        if node in required:
            return
        required.add(node)
        for source in incoming[node]:
            bfs(source)
    
    for node in outputs:
        bfs(node)
    
    return required


def feed_forward_layers(inputs: list[int], outputs: list[int], edges: list[tuple[int, int]]):
    # Want to find: layers of nodes that can be computed in parallel

    # Get required hidden nodes
    required = required_for_output(outputs, edges)  # O(|V| + |E|)

    # Create adjacency lists to speed up search
    adj = defaultdict(set)
    incoming = defaultdict(set)
    for s, t in edges:
        if (s in required or s in inputs) and t in required:
            adj[s].add(t)
            incoming[t].add(s)

    # Collect layers of nodes that can be computed in parallel
    layers = [set(inputs)]
    while True:
        layer = set()
        # For all nodes pointed to by nodes in the last layer
        for node in layers[-1]:
            for target in adj[node]:
                # Check that all incoming nodes are in previous layers
                if all(any(source in l for l in layers) for source in incoming[target]):
                    layer.add(target)
        if not layer:
            break
        layers.append(layer)

    return layers[1:] # Exclude input layer


def directed_from_input_to_output(inputs: list[int], outputs: list[int], edges: list[tuple[int, int]]):
    # Want to find: all nodes that lie on a directed path from any input node to any output node
    # (excluding paths that go through input nodes, since these are not part of the network)

    # Create adjacency list to speed up search
    adj = defaultdict(set)
    for s, t in edges:  # O(|E|)
        adj[s].add(t)

    # We can run depth-first search from each input node
    required = set(outputs)
    excluded = set(inputs)  # Input nodes are excluded

    def dfs(node, upstream=set()):
        if node in required:
            return True
        if node in excluded or node in upstream:
            # If this node is excluded, or if this node is already upstream, then we don't need to search it
            # (prevents infinite loops in the case of cycles)
            return False
        
        for target in adj[node]:
            if dfs(target, upstream | {node}):
                # If any of the nodes that this node points to are required, then this node is required
                required.add(node)
                return True
        
        # If none of the nodes that this node points to are required, then this node is not required
        excluded.add(node)
        return False
    
    for node in inputs:
        dfs(node)

    return required

