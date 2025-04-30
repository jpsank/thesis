# ALife Simulation Suite

A collection of four core experiments exploring artificial-life and computational dynamics:

1. **Reaction–Diffusion CA** (`rd.py`)  
2. **GPU-Accelerated Physics Simulation** (`physics/__main__.py` + `diffusion.cl`)  
3. **GOLEM: Game Of Life Energy-Mass** (`golem.py`)  
4. **NEAT Coevolution (Grass & Prey)** (`evo/__main__.py`)

---

## Methodology

### Reaction–Diffusion Chemistry with Virtual Cells and GRNs  
We model a two-dimensional grid as a multi-chemical reaction–diffusion medium populated by “cells” that maintain their own internal chemistry. The environment carries seven continuous species—water, carbon dioxide, oxygen, sugar, ATP, protein and a generic mitogen—that diffuse according to Fick’s law and react via a unified reaction list. Some reactions proceed slowly in the background; others are catalyzed within cells based on their internal state. Each cell occluded on a grid site holds internal channels mirroring key chemicals and abstract regulatory signals. A gene regulatory network (GRN) at each cell takes as input local environmental concentrations and its internal channels, then outputs modifications to internal and membrane-transporter channels at an ATP cost. When mitogen levels and resources exceed thresholds, cells divide—splitting internal contents and copying the GRN with mutation—while cells starved of ATP die and release their contents back into the field. This framework enables emergent gradient formation, metabolism-driven patterning, and “synthetic biology” behaviors such as differentiation-like effects and signal-triggered proliferation.

### GPU-Accelerated Physics Simulation
This experiment takes inspiration from the Biomaker CA framework ([Mordvintsev et al.](https://google-research.github.io/self-organising-systems/2023/biomaker-ca/)), which simulates a richly typed biome via simple, local cellular-automaton rules. Rather than hard-coding several cell types like they do (stem, leaf, root, seed, air, dirt, etc.), we abstract every material down to its **intermolecular force** characteristics. Each pixel's RGBA color represents a four-channel “phase” vector (solid, gas, liquid, void), and we assign each phase a **solidity** parameter that proxies bond strength—strong ionic/covalent bonds for solids, hydrogen bonds for liquids, and weak London forces for gases.

At each time step, an OpenCL kernel runs over the grid and applies two core local exchanges:

1. **Gravity-like exchange** with the pixel above/below, clamped by the neighbor’s solidity. This captures how denser or more strongly bonded materials resist buoyant motion.  
2. **Diffusive exchange** with the four orthogonal neighbors, scaled by the inverse of solidity. This encodes how weakly bonded phases (e.g. air) spread rapidly, while strongly bonded phases (e.g. stone) remain localized.

By tuning solidity values (e.g. 0.999 for rock, 0.5 for water, 0.001 for air), we can observe how the system self-organizes into lifelike material interactions—liquids pool, solids pile, and something like bubbles form when low-solidity “air” rises through the higher-solidity “water” matrix. A NumPy reference implementation on the CPU highlights the dramatic throughput advantage of GPU compute.  

This experiment unifies falling-sand dynamics and fluid behavior, which could be useful in future artificial-life models that integrate material physics with metabolism, chemotaxis, and multicellular mechanics.

### GOLEM: Game Of Life Energy–Mass  
We extend Conway’s Game of Life by equipping each cell with explicit mass and energy quantities. In each tick, standard Life birth/death rules fire first, consuming or releasing mass and energy locally. Next, an “energy-mass routing” policy network—trained separately or evolved—computes per-cell demand, and available energy is redistributed conservatively across a 3×3 neighborhood before a small diffusion and leak step smooths the field. This transforms the binary automaton into a dissipative, resource-driven system whose emergent behaviors reflect both neighborhood topology and thermodynamic constraints.

### NEAT Coevolution of Grass and Prey  
We simulate two interacting populations—autotrophic “grass” and herbivorous “prey”—on a toroidal grid. Each individual carries a genome encoding a neural controller evolved via the NEAT algorithm. At each time step, organisms sense a local RGB neighborhood representing resource and agent distributions, forward these inputs through their phenotype network, and decide actions such as movement or reproduction. Actions consume or replenish an energy budget; individuals die when energy is depleted and reproduce when it exceeds a threshold, producing mutated offspring. Predator–prey dynamics emerge as grass evolves evasive or clustered growth patterns while prey evolve foraging and avoidance strategies, illustrating coevolution in a resource-constrained environment.  

## 📂 Directory Structure

```
.
├── evo/                         # NEAT-based coevolution experiments
│   ├── nn/                      # neural network modules
│   │   ├── ffnn.py 
│   │   ├── graphs.py            # network visualization and graph functions
│   │   ├── phenotype_nn.py
│   │   └── rnn.py
│   ├── genome.py                # genome and evolutionary operators
│   ├── functions.py             # activation functions
│   ├── evolve.py                # Population class
│   ├── game1.py                 # grass–herbivore sim
│   └── config.py                # NEAT hyperparameters
│
├── physics/                     # GPU physics simulation
│   ├── diffusion.cl             # OpenCL kernel
│   ├── gump.jpg                 # input image
│   └── game3.py                 # host code (PyOpenCL)
│
├── rd.py                        # Reaction–Diffusion CA
├── golem.py                     # GOLEM: Game Of Life Energy-Mass
│
├── assemble_videos.sh           # Bash script to assemble MP4s
├── plot_metrics.py              # Python script to generate plots
├── out/                         # Auto-generated outputs
│   ├── rd/
│   │   ├── frames/
│   │   └── metrics.csv
│   ├── physics/
│   │   ├── frames/
│   │   ├── gpu_times.csv
│   │   └── cpu_times.csv
│   ├── golem/
│   │   ├── frames/
│   │   └── stats.csv
│   ├── evo/
│   │   ├── frames/
│   │   └── metrics.csv
│   ├── plots/
│   │   ├── reaction_diffusion_means.png
│   │   ├── physics_timing.png
│   │   ├── golem_summary.png
│   │   └── evo_population.png
│   └── videos/                  # MP4s assembled by `assemble_videos.sh`
│       ├── rd.mp4
│       ├── physics.mp4
│       ├── golem.mp4
│       └── evo.mp4
├── Pipfile                      # project dependencies
├── Pipfile.lock
└── README.md                    # ← you are here
```

---

## ⚙️ Installation

1. **Clone** this repository  
   ```bash
   git clone https://github.com/jpsank/thesis.git
   ```
2. **Install dependencies**  
   ```bash
   pipenv install
   pipenv shell
   ```
   or
   ```bash
   conda create -n alife python=3.11
   conda activate alife
   conda install -c conda-forge \
     numpy matplotlib pandas pyopencl torch \
     jax jaxlib networkx pygame imageio pygraphviz
   ```

3. **Ensure** you have an OpenCL driver for your GPU (for GPU physics simulation)

---

## 🚀 Running Experiments

Activate your virtualenv and then run each script as follows:

### 1. Reaction–Diffusion CA  
```bash
python rd.py \
  --width 100 \
  --height 100 \
  --dt 0.1 \
  --steps 5000 \
  --vis-interval 10 \
  --output-dir out/rd
```

### 2. GPU-Accelerated Physics Simulation  
```bash
python -m physics \
  --input-image physics/gump.jpg \
  --width 576 \
  --height 384 \
  --steps 500 \
  --compare-cpu \
  --output-dir out/physics \
  --display
```

### 3. GOLEM — Game Of Life Energy-Mass  
```bash
python golem.py \
  --size 150 \
  --steps 1000 \
  --output-dir out/golem
```

### 4. NEAT Coevolution (Grass & Prey)  
```bash
python -m evo \
  --steps 5000 \
  --save-interval 10 \
  --output-dir out/evo
```

---

## 📈 Results & Visualization

1. **Assemble Videos**  
   ```bash
   ./assemble_videos.sh
   ```
   Generates:
   ```
   out/videos/rd.mp4
   out/videos/physics.mp4
   out/videos/golem.mp4
   out/videos/evo.mp4
   ```

2. **Generate Plots**  
   ```bash
   python plot_metrics.py
   ```
   Saves figures to `out/plots/`.

3. **Direct Inspection**  
   - **Frames:** `out/<experiment>/frames/*.png`  
   - **Metrics CSV:** inspect with Pandas or Excel  
   - **Plots:** `out/plots/*.png`

---

## 📊 Consolidated Results

| Experiment                         | Key Parameters                                                      | Final Metric                                                     | Video File               | Plot File                                   |
|------------------------------------|---------------------------------------------------------------------|------------------------------------------------------------------|---------------------------|---------------------------------------------|
| **Reaction–Diffusion CA**          | `--width=100 --height=100 --dt=0.1 --steps=1000`                    | Mean SUGAR at t=1000: **0.985** (from `out/rd/metrics.csv`)                | `out/videos/rd.mp4`       | `out/plots/reaction_diffusion_means.png`    |
| **GPU Physics Simulation**         | `--width=576 --height=384 --steps=500 --compare-cpu`                | Avg GPU time/step: **1.362ms** (from `out/physics/gpu_times.csv`)            | `out/videos/physics.mp4`  | `out/plots/physics_timing.png`             |
| **GOLEM (Game Of Life E-M)**       | `--size=150 --steps=1000`                   | Live cells at t=1000: **2466** (from `out/golem/stats.csv`)               | `out/videos/golem.mp4`    | `out/plots/golem_summary.png`              |
| **NEAT Coevolution (Grass & Prey)**| `--steps=5000 --save-interval=10` | Prey population at t=5000: **0** (from `out/evo/metrics.csv`)         | `out/videos/evo.mp4`      | `out/plots/evo_population.png`             |

---

## 📖 Citation

If you use this code or the experiments in your work, please cite:

> Julian Sanker, “GPU-Accelerated Agent-Based Simulation for Cell Dynamics,” Senior Thesis, Yale University, 2025.

---

## 📝 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
