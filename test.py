# mass_energy_ca_genetic.py

import numpy as np
import matplotlib.pyplot as plt

# Grid size
H, W = 100, 100

# MLP hidden size
HIDDEN = 8
# Genome dimension: W1 (18xHIDDEN) + b1 (HIDDEN) + W2 (HIDDENx9) + b2 (9)
GENOME_DIM = 18*HIDDEN + HIDDEN + HIDDEN*9 + 9

# Genetic parameters
MUTATION_STD = 0.02

def init_genomes():
    """Initialize random genomes for every cell."""
    return np.random.randn(H, W, GENOME_DIM)

def mlp_policy(genome_vec, m_patch, e_patch):
    """
    Compute 3x3 routing kernel from one cell's genome and local patches.
    genome_vec: (GENOME_DIM,)
    m_patch, e_patch: (3,3) arrays
    returns K: (3,3)
    """
    # Flatten input
    x = np.concatenate([m_patch.flatten(), e_patch.flatten()])  # (18,)
    # Unpack genome
    idx = 0
    W1 = genome_vec[idx:idx+18*HIDDEN].reshape(18, HIDDEN); idx += 18*HIDDEN
    b1 = genome_vec[idx:idx+HIDDEN]; idx += HIDDEN
    W2 = genome_vec[idx:idx+HIDDEN*9].reshape(HIDDEN, 9); idx += HIDDEN*9
    b2 = genome_vec[idx:idx+9]  # idx += 9

    # Forward pass
    h = np.tanh(x.dot(W1) + b1)         # (HIDDEN,)
    z = h.dot(W2) + b2                  # (9,)
    # Softmax
    z = z - np.max(z)
    exp_z = np.exp(z)
    K = exp_z / exp_z.sum()
    return K.reshape(3,3)

def crossover_and_mutate(parents):
    """
    Uniform crossover among parent genome vectors + Gaussian mutation.
    parents: list of 3 genome vectors, each (GENOME_DIM,)
    returns child genome vector
    """
    child = np.empty(GENOME_DIM, dtype=float)
    for i in range(GENOME_DIM):
        # pick one of the 3 parents
        p = np.random.randint(3)
        child[i] = parents[p][i]
    # mutation
    child += np.random.randn(GENOME_DIM) * MUTATION_STD
    return child

def step(mass, energy, genomes, alpha=0.0, epsilon=0.01):
    """
    One timestep with per-cell genomes and genetic operators.
    mass, energy: (H,W) arrays
    genomes: (H,W,GENOME_DIM) array
    Returns next_mass, next_energy, next_genomes.
    """
    # Helper to wrap
    def wrap(arr):
        return np.pad(arr, ((1,1),(1,1)), mode='wrap')

    m_wrap = wrap(mass)
    e_wrap = wrap(energy)
    g_wrap = np.pad(genomes, ((1,1),(1,1),(0,0)), mode='wrap')

    new_mass = mass.copy()
    new_energy = energy.copy()
    new_genomes = genomes.copy()

    # Precompute neighbor counts
    nbr_mass = sum(m_wrap[i:(i+H), j:(j+W)]
                   for i in range(3) for j in range(3)
                   if not (i==1 and j==1))

    # Deaths
    deaths = (mass == 1) & ((nbr_mass < 2) | (nbr_mass > 3))
    new_mass[deaths] = 0
    new_energy[deaths] = 1.0
    new_genomes[deaths] = 0.0  # clear genome

    # Births
    births = (mass == 0) & (nbr_mass == 3)
    bx, by = np.where(births)
    for x,y in zip(bx,by):
        # gather exactly 3 live parents
        parents = []
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                if dx==0 and dy==0: continue
                if mass[(x+dx)%H, (y+dy)%W] == 1:
                    parents.append(genomes[(x+dx)%H, (y+dy)%W])
        if len(parents) != 3:
            continue
        # check available energy across e_wrap
        donor_coords = [(dx,dy) for dx in (-1,0,1) for dy in (-1,0,1)
                        if not (dx==0 and dy==0)
                        and mass[(x+dx)%H,(y+dy)%W]==1]
        energies = [energy[(x+dx)%H,(y+dy)%W] for dx,dy in donor_coords]
        total = sum(energies)
        if total < 1.0:
            continue
        # subtract proportionally
        for (dx,dy), e_val in zip(donor_coords, energies):
            share = e_val / total
            new_energy[(x+dx)%H, (y+dy)%W] -= share
        new_mass[x,y] = 1
        new_energy[x,y] = 0.0
        # genetics: crossover + mutate
        new_genomes[x,y] = crossover_and_mutate(parents)

    # Energy routing + optional diffusion
    next_energy = np.zeros_like(new_energy)
    for x in range(H):
        for y in range(W):
            if new_energy[x,y] <= 0:
                continue
            # local patches for (m,e)
            m_patch = m_wrap[x:x+3, y:y+3]
            e_patch = e_wrap[x:x+3, y:y+3]
            # genome for this cell
            g = new_genomes[x,y]
            K = mlp_policy(g, m_patch, e_patch)  # (3,3)
            # route energy
            amt = new_energy[x,y]
            for i,dx in enumerate((-1,0,1)):
                for j,dy in enumerate((-1,0,1)):
                    nx,ny = (x+dx)%H, (y+dy)%W
                    next_energy[nx,ny] += amt * K[i,j]
    # entropy leak
    next_energy *= (1 - epsilon)

    return new_mass, next_energy, new_genomes

# ---------------- Main Simulation ----------------

if __name__ == "__main__":
    # initialize
    mass = (np.random.rand(H,W) < 0.9).astype(float)
    energy = np.zeros((H,W), float)
    genomes = init_genomes()

    steps = 200
    plt.ion()
    fig, ax = plt.subplots(figsize=(6,6))
    for t in range(steps):
        mass, energy, genomes = step(mass, energy, genomes)
        display = mass + np.clip(energy, 0, 1)
        ax.clear()
        ax.imshow(display, cmap='hot', interpolation='nearest')
        ax.set_title(f"t = {t}")
        ax.axis('off')
        plt.pause(0.01)
    plt.ioff()
    plt.show()