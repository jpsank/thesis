import numpy as np
import matplotlib.pyplot as plt
import argparse, os, csv
import imageio

# -------------------------
# 1) Enumerate chemicals
# -------------------------
H2O, CO2, O2, SUGAR, ATP, PROTEIN, MITOGEN = range(7)
N_CHEM = 7

# -------------------------
# 2) Nameless reaction list
# -------------------------
reaction_defs = [
    {  # 0: respiration
        "inputs":  {SUGAR:1, O2:6},
        "outputs": {CO2:6, H2O:6, ATP:1},
        "bg":      0.0,
        "cost":    0.1,
    },
    {  # 1: photosynthesis
        "inputs":  {CO2:6, H2O:6},
        "outputs": {SUGAR:1, O2:6},
        "bg":      0.0,
        "cost":    0.1,
    },
    {  # 2: protein biosynthesis
        "inputs":  {SUGAR:1, ATP:1},
        "outputs": {PROTEIN:1},
        "bg":      0.0,
        "cost":    0.2,
    },
    {  # 3: protein degradation → sugar
        "inputs":  {PROTEIN:1},
        "outputs": {SUGAR:1},
        "bg":      0.0,
        "cost":    0.0,
    },
    {  # 4: mitogen synthesis
        "inputs":  {SUGAR:1, ATP:1},
        "outputs": {MITOGEN:1},
        "bg":      0.0,
        "cost":    0.2,
    },
    {  # 5: mitogen degradation → sugar
        "inputs":  {MITOGEN:1},
        "outputs": {SUGAR:1},
        "bg":      0.01,   # background decay everywhere
        "cost":    0.0,
    },
]
RCOUNT = len(reaction_defs)

# -------------------------
# 5) Diffusion coefficients
# -------------------------
D = np.zeros(N_CHEM, float)
D[H2O]     = 0.2
D[CO2]     = 1.0
D[O2]      = 1.0
D[SUGAR]   = 0.01
D[ATP]     = 0.05
D[PROTEIN] = 0.0
D[MITOGEN] = 0.05

# -------------------------
# 6) Utilities & neighbors
# -------------------------
def laplacian(g):
    return (
        np.roll(g,  1, axis=0) + np.roll(g, -1, axis=0) +
        np.roll(g,  1, axis=1) + np.roll(g, -1, axis=1) -
        4*g
    )

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

neighbors = [(1,0),(-1,0),(0,1),(0,-1)]

# -------------------------
# CLI and main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser("Reaction-Diffusion CA")
    p.add_argument("--width",       type=int,   default=100,    help="Grid width")
    p.add_argument("--height",      type=int,   default=100,    help="Grid height")
    p.add_argument("--dt",          type=float, default=0.1,    help="Time step")
    p.add_argument("--steps",       type=int,   default=1000,   help="Total iterations")
    p.add_argument("--vis-interval",type=int,   default=10,     help="Frames every N steps")
    p.add_argument("--output-dir",  type=str,   default="out/rd", help="Where to save frames & metrics")
    return p.parse_args()

def main():
    # -------------------------
    # 3) Simulation parameters
    # -------------------------
    args = parse_args()
    W, H        = args.width, args.height
    dt          = args.dt
    STEPS       = args.steps
    VIS_INTERVAL= args.vis_interval
    MITOSIS_THRESH = 1.0         # mitogen level to trigger division

    # -------------------------
    # 4) Initialize fields
    # -------------------------
    fields = np.zeros((N_CHEM, W, H), float)

    # seed resources uniformly
    fields[H2O,   :, :] = 1.0
    fields[CO2,   :, :] = 1.0
    fields[O2,    :, :] = 1.0
    fields[SUGAR, :, :] = 1.0
    fields[ATP,   :, :] = 2.0  # extra ATP

    # central blob of mitogen
    cx, cy = W//2, H//2
    xx, yy = np.ogrid[:W, :H]
    mask   = (xx-cx)**2 + (yy-cy)**2 < 10**2
    fields[MITOGEN][mask] = 2.0

    # -------------------------
    # 5) Diffusion coefficients
    # -------------------------
    D = np.zeros(N_CHEM, float)
    D[H2O]     = 0.2
    D[CO2]     = 1.0
    D[O2]      = 1.0
    D[SUGAR]   = 0.01
    D[ATP]     = 0.05
    D[PROTEIN] = 0.0
    D[MITOGEN] = 0.05

    # -------------------------
    # 7) Grid‐based cell state
    # -------------------------
    # occupancy mask
    occ = np.zeros((W, H), bool)
    # seed 30 random cells
    for _ in range(30):
        x, y = np.random.randint(W), np.random.randint(H)
        occ[x,y] = True

    # persistent channels per cell-site: M + T + R
    P = 2*N_CHEM + RCOUNT
    channels    = np.zeros((P, W, H), float)
    decay_rates = np.full((P, W, H), 0.01, float)

    # cost: only reaction channels cost ATP
    cost_coeffs = np.zeros((P, W, H), float)
    for i, rd in enumerate(reaction_defs):
        cost_coeffs[2*N_CHEM + i, :, :] = rd["cost"]

    eta = 0.2  # channel responsiveness

    # GRN weights & biases per cell-site
    W_reg = np.random.randn(P, N_CHEM+P, W, H) * 0.01
    b_reg = np.zeros((P, W, H), float)

    # -------------------------
    # 8) Color mapping for composite
    # -------------------------
    chem_colors = {
        H2O:     np.array([0.0, 0.0, 1.0]),
        CO2:     np.array([0.0, 1.0, 0.0]),
        O2:      np.array([1.0, 0.0, 0.0]),
        SUGAR:   np.array([1.0, 1.0, 0.0]),
        ATP:     np.array([0.0, 1.0, 1.0]),
        PROTEIN: np.array([1.0, 0.0, 1.0]),
        MITOGEN: np.array([1.0, 1.0, 1.0]),
    }

    # -------------------------
    # 9) Visualization setup
    # -------------------------
    plt.ion()
    fig, (ax_occ, ax_comp) = plt.subplots(1,2, figsize=(8,4))
    img_occ  = ax_occ.imshow(occ.T, cmap='gray', vmin=0, vmax=1)
    ax_occ.set_title("Cells")
    comp_img = np.zeros((W, H, 3), float)
    img_comp = ax_comp.imshow(comp_img)
    ax_comp.set_title("Chemicals")
    plt.tight_layout()

    # -------------------------
    # Metrics CSV setup
    # -------------------------
    os.makedirs(os.path.join(args.output_dir, "frames"), exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "metrics.csv")
    metrics_f    = open(metrics_path, "w", newline="")
    writer       = csv.writer(metrics_f)
    writer.writerow(
        ["t"] + [f"mean_{name}" for name in ["H2O","CO2","O2","SUGAR","ATP","PROTEIN","MITOGEN"]]
    )

    # -------------------------
    # 10) Main simulation loop
    # -------------------------
    for t in range(STEPS):
        # A) Diffuse all chemicals
        for c in range(N_CHEM):
            fields[c] += D[c] * laplacian(fields[c]) * dt

        # B) Global background reactions
        for rd in reaction_defs:
            bg = rd["bg"]
            if bg <= 0: continue
            rate  = bg * dt
            driver, _ = next(iter(rd["inputs"].items()))
            decay = fields[driver] * rate
            for in_c, s in rd["inputs"].items():
                fields[in_c] -= s * decay
            for out_c, s in rd["outputs"].items():
                fields[out_c] += s * decay

        # C) Cell‐mediated loop
        new_occ      = occ.copy()
        new_channels = channels.copy()
        new_W_reg    = W_reg.copy()
        new_b_reg    = b_reg.copy()

        xs, ys = np.nonzero(occ)
        for x,y in zip(xs, ys):
            # 1) Sense & update channels
            fv  = fields[:,x,y]
            inp = np.concatenate([fv, channels[:,x,y]])
            targ  = sigmoid(W_reg[:,:,x,y].dot(inp) + b_reg[:,x,y])
            delta = eta * (targ - channels[:,x,y])
            atp_cost = np.sum(np.abs(delta) * cost_coeffs[:,x,y]) * dt
            fields[ATP,x,y] -= atp_cost
            ch = channels[:,x,y] + delta*dt
            ch *= (1 - decay_rates[:,x,y]*dt)
            new_channels[:,x,y] = ch

            # 2) Cell‐mediated reactions
            Rvals = ch[2*N_CHEM:2*N_CHEM+RCOUNT]
            for i, rd in enumerate(reaction_defs):
                rate = max(0.0, rd["bg"] + Rvals[i]) * dt
                fields[ATP,x,y] -= abs(Rvals[i]) * rd["cost"] * dt
                if all(fields[in_c,x,y] >= s*rate for in_c,s in rd["inputs"].items()):
                    for in_c,s in rd["inputs"].items():
                        fields[in_c,x,y] -= s*rate
                    for out_c,s in rd["outputs"].items():
                        fields[out_c,x,y] += s*rate

            # 3) Transport (free)
            T = ch[N_CHEM:2*N_CHEM]
            for i in range(N_CHEM):
                flow = T[i] * fields[i,x,y] * dt
                fields[i,x,y] -= flow
                fields[i,x,y] += flow

            # 4) Movement (free)
            M = ch[:N_CHEM]
            best_score, best_dir = 0.0, None
            for dx,dy in neighbors:
                nx,ny = (x+dx)%W, (y+dy)%H
                grad = fields[:,nx,ny] - fields[:,x,y]
                score = M.dot(grad)
                if score > best_score:
                    best_score, best_dir = score, (nx,ny)
            if best_dir:
                nx, ny = best_dir
                new_occ[nx,ny] = True
                new_occ[x,y]    = False
                new_channels[:,nx,ny] = new_channels[:,x,y]
                new_W_reg[:,:,nx,ny]  = new_W_reg[:,:,x,y]
                new_b_reg[:,   nx,ny] = new_b_reg[:,   x,y]

            # 5) Mitosis via mitogen threshold
            if fields[MITOGEN,x,y] >= MITOSIS_THRESH:
                empties = [((x+dx)%W,(y+dy)%H) for dx,dy in neighbors
                        if not new_occ[(x+dx)%W,(y+dy)%H]]
                if empties:
                    nx,ny = empties[np.random.randint(len(empties))]
                    local = fields[:,x,y].copy()
                    fields[:, x, y]   = local * 0.5
                    fields[:, nx, ny] = local * 0.5

                    new_occ[nx,ny]         = True
                    new_channels[:,nx,ny]  = new_channels[:,x,y]
                    new_W_reg[:,:,nx,ny]   = new_W_reg[:,:,x,y] + np.random.randn(P,N_CHEM+P)*1e-3
                    new_b_reg[:,   nx,ny]  = new_b_reg[:,   x,y]    + np.random.randn(P)*1e-3

            # 6) Death if no ATP
            if fields[ATP,x,y] < 0.1:
                new_occ[x,y] = False

        # commit
        occ       = new_occ
        channels  = new_channels
        W_reg     = new_W_reg
        b_reg     = new_b_reg

        # log metrics
        means = [round(float(fields[c].mean()), 6) for c in range(N_CHEM)]
        writer.writerow([t] + means)

        # D) Save visualization frames
        if t % VIS_INTERVAL == 0:
            img_occ.set_data(occ.T.astype(float))
            ax_occ.set_title(f"Cells (t={t})")

            comp = np.zeros((W, H, 3), float)
            for chem, color in chem_colors.items():
                comp += fields[chem].T[..., None] * color[None, None, :]
            comp /= comp.max() if comp.max()>0 else 1
            comp = np.clip(comp, 0, 1)
            img_comp.set_data(comp)
            ax_comp.set_title("Chemicals")

            # save frame
            out_path = os.path.join(args.output_dir, "frames", f"frame_{t:05d}.png")
            fig.savefig(out_path, dpi=fig.dpi, bbox_inches='tight')
            plt.pause(0.01)

    # cleanup
    metrics_f.close()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
