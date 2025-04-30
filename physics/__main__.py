# physics_sim.py

import argparse
import os
import time
import numpy as np
import pyopencl as cl
from PIL import Image
import pygame
import imageio

# Constants from diffusion.cl
offsets = [(1,0), (-1,0), (0,1), (0,-1)]
solidities      = np.array([0.999, 0.5/800., 0.5,    0.0   ], dtype=np.float32)
solidities_inv  = 1.0 - solidities

def vmax(v):
    return np.maximum.reduce(v[..., :3], axis=-1)

def compute_gravity(north, south):
    a = (1.0 - south) * solidities
    b = vmax(south * solidities)[..., None]
    return np.minimum(np.maximum(a - b, 0.0), north)

def compute_diffusion(pixel, neighbor):
    term1 = (neighbor - pixel) * solidities_inv
    term2 = (vmax(neighbor) - vmax(pixel))[..., None]
    diff = term1 + term2
    return 0.2 * np.minimum(np.maximum(diff, -pixel), neighbor)

def cpu_apply_physics(input_img):
    # ensure float32 everywhere
    input_img = input_img.astype(np.float32)
    H, W, C   = input_img.shape
    out       = input_img.copy()

    # Gravity
    north = np.zeros_like(input_img, dtype=np.float32)
    south = np.zeros_like(input_img, dtype=np.float32)
    north[1:,:,:]  = input_img[:-1,:,:]
    south[:-1,:,:] = input_img[1:,:,:]

    g_north = compute_gravity(north, input_img).astype(np.float32)
    g_south = compute_gravity(input_img, south).astype(np.float32)
    out     = out + g_north - g_south

    # Diffusion
    for dx, dy in offsets:
        shifted = np.zeros_like(input_img, dtype=np.float32)
        sx = slice(max(dx,0), H + min(dx,0))
        sy = slice(max(dy,0), W + min(dy,0))
        tx = slice(max(-dx,0), H + min(-dx,0))
        ty = slice(max(-dy,0), W + min(-dy,0))
        shifted[tx,ty] = input_img[sx,sy]
        diff = compute_diffusion(input_img, shifted).astype(np.float32)
        out += diff

    return out

def parse_args():
    p = argparse.ArgumentParser("GPU-Accelerated Physics Simulation")
    p.add_argument("--input-image", type=str,   default="physics/gump.jpg",
                   help="Path to input image")
    p.add_argument("--width",       type=int,   default=576,
                   help="Kernel grid width (must match image resize)")
    p.add_argument("--height",      type=int,   default=384,
                   help="Kernel grid height (must match image resize)")
    p.add_argument("--steps",       type=int,   default=500,
                   help="Number of simulation steps")
    p.add_argument("--compare-cpu", action="store_true",
                   help="Also run a CPU physics for timing comparison")
    p.add_argument("--kernel",      type=str,   default="physics/diffusion.cl",
                   help="Path to OpenCL kernel source")
    p.add_argument("--output-dir",  type=str,   default="out/physics",
                   help="Directory to save frames and CSVs")
    p.add_argument("--display",     action="store_true",
                   help="Show Pygame window during simulation")
    return p.parse_args()

def main():
    args = parse_args()

    # Prepare output directories
    frames_dir = os.path.join(args.output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    gpu_csv_path = os.path.join(args.output_dir, "gpu_times.csv")
    gpu_f = open(gpu_csv_path, "w", newline="")
    gpu_f.write("step,gpu_ms\n")
    if args.compare_cpu:
        cpu_csv_path = os.path.join(args.output_dir, "cpu_times.csv")
        cpu_f = open(cpu_csv_path, "w", newline="")
        cpu_f.write("step,cpu_ms\n")

    # Optional Pygame display
    if args.display:
        pygame.init()
        screen = pygame.display.set_mode((args.width, args.height))

    # OpenCL setup
    platform = cl.get_platforms()[0]
    device   = platform.get_devices()[0]
    ctx      = cl.Context([device])
    queue    = cl.CommandQueue(ctx)

    # Load and build kernel
    with open(args.kernel) as f:
        kernel_src = f.read()
    program = cl.Program(ctx, kernel_src).build()
    kernel = program.apply_physics

    # Load input image
    im = Image.open(args.input_image).convert("RGBA").resize((args.width, args.height))
    input_data = np.array(im).astype(np.float32) / 255.0
    input_data = np.clip(input_data - 0.5, 0, 1)

    H, W = args.height, args.width
    mf = cl.mem_flags
    in_buf  = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=input_data)
    out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, input_data.nbytes)

    # Simulation loop
    for step in range(args.steps):
        # GPU step
        t0 = time.perf_counter()
        kernel.set_args(in_buf, out_buf, np.uint32(W), np.uint32(H))
        cl.enqueue_nd_range_kernel(queue, kernel, (W, H), None)
        queue.finish()
        t1 = time.perf_counter()
        gpu_ms = (t1 - t0) * 1000
        gpu_f.write(f"{step},{gpu_ms:.3f}\n")

        # Retrieve GPU output
        output_data = np.empty_like(input_data)
        cl.enqueue_copy(queue, output_data, out_buf)
        queue.finish()
        in_buf = out_buf
        out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, input_data.nbytes)

        # CPU comparison
        if args.compare_cpu:
            t0c = time.perf_counter()
            cpu_out = cpu_apply_physics(input_data)
            t1c = time.perf_counter()
            cpu_ms = (t1c - t0c) * 1000
            cpu_f.write(f"{step},{cpu_ms:.3f}\n")
            input_data = output_data
            # Optional correctness check:
            # if not np.allclose(cpu_out, output_data, atol=1e-3, rtol=1e-3):
            #     max_err = np.max(np.abs(cpu_out - output_data))
            #     print(f"Warning: max CPU↔GPU diff = {max_err:.5f}")

        # Save frame
        rgb_frame = (output_data[..., :3] * 255).astype(np.uint8)
        frame_path = os.path.join(frames_dir, f"frame_{step:05d}.png")
        imageio.imsave(frame_path, rgb_frame)

        # Display if requested
        if args.display:
            surf = pygame.surfarray.make_surface(rgb_frame.swapaxes(0,1))
            screen.blit(surf, (0,0))
            pygame.display.flip()
            for evt in pygame.event.get():
                if evt.type == pygame.QUIT:
                    args.steps = step + 1

    # Cleanup
    gpu_f.close()
    if args.compare_cpu:
        cpu_f.close()
    if args.display:
        pygame.quit()

if __name__ == "__main__":
    main()