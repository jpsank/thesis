# ALife Simulation Suite

A collection of four core experiments exploring artificial-life and computational dynamics:

1. **Reaction–Diffusion CA** (`rd.py`)  
2. **GPU-Accelerated Physics Simulation** (`physics/__main__.py` + `diffusion.cl`)  
3. **GOLEM: Game Of Life Energy-Mass** (`golem.py`)  
4. **NEAT Coevolution (Grass & Prey)** (`evo/__main__.py`)

---

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
