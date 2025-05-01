# ALife Simulation Suite

A collection of four core experiments exploring artificial-life and computational dynamics:

1. **Reaction–Diffusion CA** (`rd.py`)  
2. **GPU-Accelerated Physics Simulation** (`physics/__main__.py` + `diffusion.cl`)  
3. **GoLEM: Game Of Life with Energy-Mass equivalence** (`golem.py`)  
4. **NEAT Coevolution (Grass & Prey)** (`evo/__main__.py`)

---

## Methodology

### Reaction–Diffusion Chemistry with Virtual Cells & GRNs

Building on agent-based models of cancer and immune dynamics (e.g. the Krishnaswamy Lab’s EuclideanSimulation), this experiment abstracts away biochemical and cell-type specifics to focus on fundamental life principles. The two-dimensional grid carries seven continuous species—water, CO₂, O₂, sugar, ATP, protein, and a generic mitogen—that diffuse per Fick’s law and react via a unified reaction list with both background and cell-catalyzed rates.  

Each cell occupies one grid site and maintains **three functional channels** for every chemical:  
1. **Reaction channel** (enzymes): positive values catalyze production reactions; negative values catalyze consumption.  
2. **Transport channel** (transporters): controls active import or export against gradients.  
3. **Motor channel** (motors): dictates energy-driven movement toward or away from chemical cues.  

Channels are driven by a small, evolvable neural network that takes environmental concentrations and a cell’s current channels as inputs and outputs channel update values at an ATP cost—abstracting gene-to-protein translation and receptor dynamics into a unified, energy-conscious decision process. When mitogen and resource thresholds are met, cells divide—splitting chemical resources and mutating network weights—while ATP depletion or lifespan limits trigger death and release of internal contents back into the field.  

This minimal yet biologically inspired framework emphasizes **metabolism**, **chemotaxis**, **transport**, and **division** as the core capabilities of life, enabling emergent gradient formation, differentiation-like cycles, and synthetic-biology behaviors without hard-coding specialized cell types.  

### GPU-Accelerated Physics Simulation
This experiment takes inspiration from the Biomaker CA framework ([Randazzo et al.](https://google-research.github.io/self-organising-systems/2023/biomaker-ca/)), which simulates a richly typed biome via simple, local cellular-automaton rules. Rather than hard-coding several cell types like they do (stem, leaf, root, seed, air, dirt, etc.), we abstract every material down to its **intermolecular force** characteristics. Each pixel's RGBA color represents a four-channel “phase” vector (solid, gas, liquid, void), and we assign each phase a **solidity** parameter that proxies bond strength—strong ionic/covalent bonds for solids, hydrogen bonds for liquids, and weak London forces for gases.

At each time step, an OpenCL kernel runs over the grid and applies two core local exchanges:

1. **Gravity-like exchange** with the pixel above/below, clamped by the neighbor’s solidity. This captures how denser or more strongly bonded materials resist buoyant motion.  
2. **Diffusive exchange** with the four orthogonal neighbors, scaled by the inverse of solidity. This encodes how weakly bonded phases (e.g. air) spread rapidly, while strongly bonded phases (e.g. stone) remain localized.

By tuning solidity values (e.g. 0.999 for rock, 0.5 for water, 0.001 for air), we can observe how the system self-organizes into lifelike material interactions—liquids pool, solids pile, and something like bubbles form when low-solidity “air” rises through the higher-solidity “water” matrix. A NumPy reference implementation on the CPU highlights the dramatic throughput advantage of GPU compute.  

This experiment unifies falling-sand dynamics and fluid behavior, which could be useful in future artificial-life models that integrate material physics with metabolism, chemotaxis, and multicellular mechanics.

### GoLEM: Game Of Life with Energy–Mass equivalence  
This experiment reimagines Conway’s Game of Life as a miniature physics sandbox governed by conservation laws and entropy. Inspired by Einstein’s $E=mc^2$, we treat mass and energy as interconvertible yet globally conserved quantities: a cell birth consumes one unit of energy, and a cell death releases that energy back into the field. In the absence of life, the system tends toward “entropy”–increasing equilibrium under standard Life rules.  

To capture how living patterns locally decrease entropy, each cell carries both mass and energy stores and participates in a learned, decentralized energy‐routing process. We augment the binary GoL grid with a continuous energy channel plus a set of hidden “genome” channels. A neural‐policy CA—analogous to [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/)—reads local mass, energy, and hidden channels, then directs energy flows across each 3×3 neighborhood, conserving the total and modeling how organisms “hijack” physics to persist ordered patterns. Cells inherit their parents' crossed-over policy parameters upon reproduction and lose them at death, enabling evolutionary adaptation of energy‐management strategies.  

By fusing Life’s simple birth/death topology with energy–mass conversion, entropy considerations, and an evolvable neural CA policy, GoLEM aims to exhibit richer emergent phenomena—sustained oscillators, self-organized energy waves, and resource-driven pattern formation—that mirror how real organisms defy equilibrium to survive and replicate under thermodynamic constraints.  

### NEAT Coevolution of Grass and Prey

In this experiment, we simulate an abstract ecosystem of autotrophic “grass” and herbivorous “prey” on a toroidal grid, with both species’ behaviors governed by evolvable neural controllers optimized via the NEAT algorithm. Each organism carries a genome that encodes a Compositional Pattern-Producing Network (CPPN), which in turn generates a phenotype neural network mapping local sensory inputs to action outputs.

- **Sensory Inputs:**  
  - Grass senses a **3×3** RGB patch of its immediate neighborhood to decide where to propagate or expend energy.  
  - Prey senses a larger **9×9** RGB field, enabling more sophisticated foraging and avoidance strategies over a wider area.

- **Actions:**  
  - **Movement:** networks output Δx, Δy vectors; prey use theirs to move toward nutrients or away from threats, consuming energy proportional to distance moved.  
  - **Asexual Reproduction:** when energy exceeds a threshold, an organism can clone itself into an adjacent empty cell.  
  - **Sexual Reproduction (Prey only):** if two prey mutually point at each other (each network’s Δx, Δy directs toward the other’s location) and both have sufficient energy, they produce a shared offspring in a neighboring empty site, combining and mutating both genomes.

- **Energy Dynamics:**  
  - Grass regenerates energy passively over time and must maintain a minimum to reproduce.  
  - Prey gain energy by consuming grass when moving into a grass-occupied cell, and lose energy via movement and reproduction. Death occurs if energy falls to zero.

By coevolving these simple sensory-motor networks, grasses adapt growth patterns that reduce predation risk, while prey evolve foraging tactics—clustering, dispersal, ambush—and even mutual breeding behaviors. Over thousands of generations, this setup yields rich ecological dynamics: spatial patchiness in grass, prey herding and avoidance, and arms-race feedback loops, illustrating how minimal neural controllers under selection can generate complex, lifelike interactions.  

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
├── golem.py                     # GoLEM: Game Of Life Energy-Mass
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

### 3. GoLEM — Game Of Life Energy-Mass  
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
| **GoLEM (Game Of Life E-M)**       | `--size=150 --steps=1000`                   | Live cells at t=1000: **2466** (from `out/golem/stats.csv`)               | `out/videos/golem.mp4`    | `out/plots/golem_summary.png`              |
| **NEAT Coevolution (Grass & Prey)**| `--steps=5000 --save-interval=10` | Prey population at t=5000: **0** (from `out/evo/metrics.csv`)         | `out/videos/evo.mp4`      | `out/plots/evo_population.png`             |

---

## 📖 Citation

If you use this code or the experiments in your work, please cite:

> Julian Sanker, “GPU-Accelerated Agent-Based Simulation for Cell Dynamics,” Senior Thesis, Yale University, 2025.

---

## 📝 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
