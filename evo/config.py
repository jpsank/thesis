import numpy as np

np.random.seed(0)

# Structural mutations
MUTATE_ADD_NODE = 0.1
MUTATE_ADD_EDGE = 0.2
MUTATE_REMOVE_NODE = 0.02
MUTATE_REMOVE_EDGE = 0.05

# Point mutations
MUTATE_ACT_FUNC = 0.1
MUTATE_BIAS = 0.5
MUTATE_BIAS_SCALE = 0.1
MUTATE_WEIGHT = 0.5
MUTATE_WEIGHT_SCALE = 0.1

ACTIVATIONS = ["sigmoid", "relu", "tanh", "sine", "cosine"]

# Speciation
DISJOINT_COEFF = 1.0
SPECIES_THRESHOLD = 3.0
