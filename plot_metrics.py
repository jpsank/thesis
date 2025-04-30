import pandas as pd
import os
import matplotlib.pyplot as plt

# Ensure output directory exists
PLOTS_DIR = "out/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# 1) Reaction–Diffusion: mean chemical concentrations
try:
    rd = pd.read_csv('out/rd/metrics.csv')
    plt.figure()
    for col in rd.columns[1:]:
        plt.plot(rd['t'], rd[col], label=col)
    plt.xlabel('Timestep')
    plt.ylabel('Mean Concentration')
    plt.legend()
    plt.title('Reaction–Diffusion Mean Concentrations')
    plt.savefig(os.path.join(PLOTS_DIR, 'reaction_diffusion_means.png'))
    plt.close()
except FileNotFoundError:
    print("Skip RD plot: out/rd/metrics.csv not found")

# 2) Physics Simulation: GPU vs CPU timings
try:
    phys_gpu = pd.read_csv('out/physics/gpu_times.csv')
    plt.figure()
    plt.plot(phys_gpu['step'], phys_gpu['gpu_ms'], label='GPU')
    try:
        phys_cpu = pd.read_csv('out/physics/cpu_times.csv')
        plt.plot(phys_cpu['step'], phys_cpu['cpu_ms'], label='CPU')
    except FileNotFoundError:
        pass
    plt.xlabel('Step')
    plt.ylabel('Time (ms)')
    plt.legend()
    plt.title('Physics Simulation Timing')
    plt.savefig(os.path.join(PLOTS_DIR, 'physics_timing.png'))
    plt.close()
except FileNotFoundError:
    print("Skip Physics plot: out/physics/gpu_times.csv not found")

# Combined GOLEM figure
try:
    golem = pd.read_csv('out/golem/stats.csv')
    plt.figure(figsize=(8, 5))
    plt.plot(golem['t'], golem['total_mass'], label='Total Mass')
    plt.plot(golem['t'], golem['total_energy'], label='Total Energy')
    plt.plot(golem['t'], golem['live_cells'], label='Live Cells')
    plt.xlabel('Timestep')
    plt.ylabel('Value')
    plt.title('GOLEM Metrics Over Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'golem_summary.png'))
    plt.close()
except FileNotFoundError:
    print("GOLEM stats.csv not found.")

# 4) Evolutionary Simulation: population sizes
try:
    evo = pd.read_csv('out/evo/metrics.csv')
    plt.figure()
    plt.plot(evo['t'], evo['grass_pop'], label='Grass Pop')
    plt.plot(evo['t'], evo['prey_pop'], label='Prey Pop')
    plt.xlabel('Timestep')
    plt.ylabel('Population')
    plt.legend()
    plt.title('Evolution Simulation Populations')
    plt.savefig(os.path.join(PLOTS_DIR, 'evo_population.png'))
    plt.close()
except FileNotFoundError:
    print("Skip Evo plot: out/evo/metrics.csv not found")