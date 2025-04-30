import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os, csv
import imageio

def get_args():
    p = argparse.ArgumentParser("GOLEM — Game Of Life Energy-Mass CA")
    p.add_argument("--steps",       type=int,   default=1000,  help="Number of CA iterations")
    p.add_argument("--size",        type=int,   default=150,   help="Grid height/width")
    # p.add_argument("--init-pattern",type=str,   default="random",
    #                choices=["random","gosper","r-pentomino"],
    #                help="Initial seed pattern")
    # p.add_argument("--energy",      type=float, default=5.0,   help="Initial energy per live cell")
    p.add_argument("--output-dir",  type=str,   default="out/golem",
                   help="Where to dump PNG frames & stats.csv")
    return p.parse_args()


class EnergyNCA(nn.Module):
    def __init__(self, genome_dim, hidden=32):
        super().__init__()
        # Perception conv: wrap‐around
        self.percep = nn.Conv2d(
            in_channels=2, out_channels=4,
            kernel_size=3, padding=1,
            bias=False, padding_mode='circular'
        )
        with torch.no_grad():
            w = torch.zeros(4, 2, 3, 3)
            # identity
            w[0,0,1,1] = 1.0
            # neighbor‐sum
            w[1,0,:,:] = 1.0
            w[1,0,1,1] = 0.0
            # Sobel X
            w[2,0] = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32)
            # Sobel Y
            w[3,0] = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32)
            self.percep.weight.copy_(w)

        # update MLP
        self.update1 = nn.Conv2d(4 + 1 + genome_dim, hidden, kernel_size=1)
        # single‐channel demand output
        self.update2 = nn.Conv2d(hidden, 1, kernel_size=1)

    def forward(self, mass, energy, genome):
        # mass, energy: (B,1,H,W); genome: (B,G,H,W)
        x      = torch.cat([mass, energy], dim=1)        # (B,2,H,W)
        p      = self.percep(x)                          # (B,4,H,W)
        u      = torch.cat([p, energy, genome], dim=1)   # (B,5+G,H,W)
        h      = F.relu(self.update1(u))                 # (B,hidden,H,W)
        return self.update2(h)                           # (B,1,H,W)


def ca_step(mass, energy, genome,
            policy, neighbor_conv, conv_demand,
            leak=0.01, sigma=0.1, diff_rate=0.01, eps=1e-6):
    B, _, H, W = mass.shape
    _, Gp1, _, _ = genome.shape  # G+1 for mutation‐rate
    offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]

    # 1) birth/death masks
    nbr    = neighbor_conv(mass)
    deaths = (mass==1) & ((nbr<2)|(nbr>3))
    births = (mass==0) & (nbr==3)

    new_mass   = mass.clone()
    new_energy = energy.clone()
    new_genome = genome.clone()

    # 2) apply deaths
    new_energy[deaths] += new_mass[deaths]
    new_mass[deaths]    = 0

    # 3) handle births (energy draw + genome inherit)
    coords = births.nonzero(as_tuple=False)
    for b, _, x, y in coords:
        parents = []
        for dx, dy in offsets:
            if dx==0 and dy==0: continue
            xi, yi = (x+dx)%H, (y+dy)%W
            if mass[b,0,xi,yi] == 1:
                parents.append((xi, yi))
        if len(parents) != 3:
            raise ValueError(f"Expected 3 parents, got {len(parents)}")

        vals  = [new_energy[b,0,xi,yi].item() for xi, yi in parents]
        total = sum(vals)
        if total < 1.0:
            continue
        for (xi, yi), v in zip(parents, vals):
            share = v/total
            new_energy[b,0,xi,yi] -= share

        # crossover + mutation
        pg      = torch.stack([genome[b,:,xi,yi] for xi,yi in parents], dim=0)  # (3,G+1)
        chooser = torch.randint(0,3,(Gp1,), device=genome.device)
        child   = pg[chooser, torch.arange(Gp1, device=genome.device)]
        parent_rates = pg[:, -1]
        mut_probs    = parent_rates[chooser]
        mask         = torch.rand(Gp1, device=genome.device) < mut_probs
        noise        = torch.randn(Gp1, device=genome.device) * sigma
        child = child + mask.float() * noise

        new_genome[b,:,x,y] = child
        new_mass[b,0,x,y]  = 1

    # 1) raw signed demand
    raw = policy(new_mass, new_energy, new_genome).squeeze(1)  # (B,H,W)

    # 2) global shift so demand ≥ 0
    #    subtract the minimum over each batch-instance
    batch_min = raw.view(B, -1).min(dim=1)[0].view(B,1,1)  # (B,1,1)
    d_shift   = raw - batch_min                          # now >= 0 everywhere

    # 3) sum over 3×3 neighborhoods
    total_dem = conv_demand(d_shift.unsqueeze(1))\
                   .squeeze(1)\
                   .clamp(min=eps)  # (B,H,W)

    # 4) redistribute each cell’s energy conservatively
    e_old = new_energy[:,0]         # (B,H,W)
    e2    = torch.zeros_like(e_old)
    for dx, dy in [(-1,-1),(-1,0),(-1,1),
                   ( 0,-1),( 0,0),( 0,1),
                   ( 1,-1),( 1,0),( 1,1)]:
        shifted = torch.roll(d_shift, shifts=(-dx,-dy), dims=(1,2))
        w       = shifted / total_dem
        transfer= e_old * w
        e2     += torch.roll(transfer, shifts=(dx,dy), dims=(1,2))

    new_energy = e2.unsqueeze(1)

    # 6) diffusion
    nbr_e      = neighbor_conv(new_energy)
    avg_e      = nbr_e / 8.0
    new_energy = new_energy*(1-diff_rate) + avg_e*diff_rate

    # 7) clamp + leak
    new_energy = new_energy * (1 - leak)

    return new_mass, new_energy, new_genome


if __name__ == '__main__':
    args = get_args()
    B, H, W = 1, args.size, args.size
    G       = 16
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(os.path.join(args.output_dir, "frames"), exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "stats.csv")
    csvfile = open(metrics_path, "w", newline="")
    writer  = csv.writer(csvfile)
    writer.writerow(["t", "total_mass", "total_energy", "total_conserved", "live_cells"])

    # neighbor conv for birth/death & diffusion (circular)
    neighbor_conv = nn.Conv2d(
        in_channels=1, out_channels=1,
        kernel_size=3, padding=1,
        bias=False, padding_mode='circular'
    ).to(device)
    k = torch.ones(1,1,3,3,device=device)
    k[0,0,1,1] = 0
    neighbor_conv.weight.data = k
    neighbor_conv.weight.requires_grad = False

    # conv for summing demands in 3×3 (circular)
    conv_demand = nn.Conv2d(
        in_channels=1, out_channels=1,
        kernel_size=3, padding=1,
        bias=False, padding_mode='circular'
    ).to(device)
    kernel       = torch.ones(1,1,3,3,device=device)
    conv_demand.weight.data = kernel
    conv_demand.weight.requires_grad = False

    policy  = EnergyNCA(genome_dim=G+1, hidden=32).to(device)

    mass    = (torch.rand(B,1,H,W,device=device) < 0.2).float()
    energy  = torch.zeros(B,1,H,W,device=device)
    genes   = torch.randn(B,G,H,W,device=device) * 0.5
    mr      = torch.full((B,1,H,W), 0.05, device=device)
    genome  = torch.cat([genes, mr], dim=1)

    plt.ion()
    fig, ax = plt.subplots(figsize=(5,5))
    # log metrics
    total_mass   = float(mass.sum().item())
    total_energy = float(energy.sum().item())
    total_conserved = total_mass + total_energy
    live_cells   = int(mass.sum().item())
    writer.writerow([-1, total_mass, total_energy, total_conserved, live_cells])
    for t in range(args.steps):
        mass, energy, genome = ca_step(
            mass, energy, genome,
            policy, neighbor_conv, conv_demand,
            leak=0.0, sigma=0.1, diff_rate=0.05
        )
        
        # log metrics
        total_mass   = float(mass.sum().item())
        total_energy = float(energy.sum().item())
        total_conserved = total_mass + total_energy
        live_cells   = int(mass.sum().item())
        writer.writerow([t, total_mass, total_energy, total_conserved, live_cells])

        # check conservation
        if not np.isclose(mass.sum().item() + energy.sum().item(), total_conserved, atol=1):
            raise ValueError("Mass+energy not conserved barring rounding errors!")
        if energy.min() < 0:
            raise ValueError("Energy went negative!")
        
        # custom color display:
        e = energy[0,0].cpu().detach().numpy().clip(0,1)
        m = mass[0,0].cpu().detach().numpy()
        rgb = np.zeros((H, W, 3), dtype=np.float32)
        # live cells: black → red by energy
        # dead cells: white → yellow by energy
        rgb[...,0] = (1-m) + m * e            # R = 1 if dead, = e if live
        rgb[...,1] = (1-m)                     # G = 1 if dead, = 0 if live
        rgb[...,2] = (1-m) * (1 - e)           # B = 1-e if dead, = 0 if live

        ax.clear()
        ax.imshow(rgb, interpolation='nearest')
        ax.set_title(f't = {t}')
        ax.axis('off')

        # save a colorized frame every N steps
        if t % 10 == 0:
            # save via imageio
            out_path = os.path.join(args.output_dir, "frames", f"frame_{t:05d}.png")
            imageio.imsave(out_path, (rgb*255).astype(np.uint8))

        plt.pause(0.0001)

    plt.ioff()
    plt.show()
    csvfile.close()
    print(f"Saved metrics to {metrics_path}")
    print("Done.")