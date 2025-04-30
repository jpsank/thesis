import pyopencl as cl
from PIL import Image
import numpy as np
import pygame

input_size = 128
output_size = 64

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Set up OpenCL context and queue
platforms = cl.get_platforms()
platform = platforms[0]  # Choose the first platform
devices = platform.get_devices(cl.device_type.ALL)
device = devices[0]  # Choose the first device
context = cl.Context([device])
queue = cl.CommandQueue(context)

# Load image and create buffers
image = Image.open('gump.jpg').convert('RGBA').resize((WIDTH, HEIGHT))
image_data = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]
image_data = np.clip(image_data - 0.5, 0, 1)  # Make image darker
height, width = image_data.shape[:2]

input_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=image_data)
output_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, image_data.nbytes)

# Kernel code to apply diffusion to 4-neighbors
with open('diffusion.cl', 'r') as f:
    kernel_code = f.read()

# Build and run the kernel
program = cl.Program(context, kernel_code).build()
kernel = program.apply_physics

clock = pygame.time.Clock()
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    kernel.set_args(input_buffer, output_buffer, np.uint32(width), np.uint32(height))
    global_work_size = (width, height)
    cl.enqueue_nd_range_kernel(queue, kernel, global_work_size, None)
    queue.finish()

    # Retrieve and save the output
    output_data = np.empty_like(image_data)
    cl.enqueue_copy(queue, output_data, output_buffer)
    queue.finish()
    print(np.max(output_data))
    input_buffer = output_buffer
    output_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, image_data.nbytes)

    output_image = (255 * output_data[:, :, :3]).swapaxes(0, 1).astype(np.uint8)
    pygame_image = pygame.surfarray.make_surface(output_image)

    screen.blit(pygame_image, (0, 0))
    pygame.display.flip()
    clock.tick(120)

# Clean up
input_buffer.release()
output_buffer.release()
pygame.quit()
