from dataclasses import dataclass
import numpy as np
import pygame
from genome import Genome, Node, Edge
from nn.rnn import RecurrentNetwork
from nn.phenotype_nn import PhenotypeNetwork, RecurrentPhenotypeNetwork
from nn.ffnn import FeedForwardNetwork
from evolve import Population
from config import *


GRASS_POP_SIZE = 20
PREY_POP_SIZE = 100
NROWS = 100
NCOLS = 100

GRASS_SUBSTRATE = [
    (3, 3, 3),
    (2, 1, 1)
] # 3x3 neighborhood, 3 channels (r, g, b), 2-vector (breed)
PREY_SUBSTRATE = [
    (9, 9, 3),
    (3, 3, 3),
    (4, 1, 1)
] # 9x9 neighborhood, 3 channels (r, g, b), hidden tensor, 4-vector (move, breed)

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
PURPLE = (255, 0, 255)
GREY = (128, 128, 128)


@dataclass
class Organism:
    id: int
    genome: Genome
    energy: float = 0.0
    cppn: FeedForwardNetwork = None
    phenotype: PhenotypeNetwork = None
    coord: tuple[int, int] = None
    color: tuple[int, int, int] = GREY
    age: int = 0

    def __post_init__(self):
        # Create feed forward neural network
        self.cppn = FeedForwardNetwork.from_genome(self.genome)


@dataclass
class Grass(Organism):
    def __post_init__(self):
        super().__post_init__()

        # Grass-specific attributes
        self.energy = 5.0
        self.color = BLUE

        # Create phenotype network
        self.phenotype = PhenotypeNetwork.create(self.cppn, GRASS_SUBSTRATE, "tanh")


@dataclass
class Prey(Organism):
    def __post_init__(self):
        super().__post_init__()

        # Prey-specific attributes
        self.energy = 50.0
        self.color = RED

        # Create phenotype network
        self.phenotype = PhenotypeNetwork.create(self.cppn, PREY_SUBSTRATE, "tanh")


def pair_breed(breeding: dict[int, tuple[int, int]], organisms: dict[int, Organism], sparse: dict[tuple[int, int], Organism]):
    # Handle organism breeding
    to_breed = set(breeding.keys())
    while to_breed:
        id1 = np.random.choice(list(to_breed))
        org1 = organisms[id1]
        r, c = breeding[id1]
        if (r, c) != org1.coord:
            # Organism wants to breed with another organism of the same type
            org2 = sparse.get((r, c))
            if org2 is not None and type(org2) == type(org1):
                # Other organism exists
                if org2.id in to_breed:
                    # Other organism wants to breed
                    r2, c2 = breeding[org2.id]
                    if (r2, c2) == org1.coord:
                        # Other organism wants to breed with this organism
                        # Then, check if there is space for child organism
                        r3, c3 = org1.coord
                        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
                        while neighbors:
                            i, j = neighbors.pop(np.random.randint(len(neighbors)))
                            r4, c4 = r3 + i, c3 + j
                            r4 %= NROWS
                            c4 %= NCOLS
                            if (r3, c3) not in sparse:
                                # Space for child organism
                                r3, c3 = r4, c4
                                break
                        if (r3, c3) != org1.coord:
                            # Reproduce
                            print("SEX")
                            yield (id1, org2.id, r3, c3)

                            # Remove parents from breeding
                            to_breed.remove(id1)
                            to_breed.remove(org2.id)
                            continue
        
        # Check if space for child organism is available
        if (r, c) not in sparse:
            # Space for child organism is available
            # Asexual reproduction
            yield (id1, id1, r, c)

        # Remove parent from breeding
        to_breed.remove(id1)


def clip(x, lo, hi):
    return max(lo, min(x, hi))


if __name__ == "__main__":
    # Create initial random populations
    print("Generating grass population...")
    pop_grass = Population.random(GRASS_POP_SIZE, 8, 2)
    grasses: dict[int, Grass] = {i: Grass(i, genome) for i, genome in pop_grass.genomes.items()}
    print("Generating prey population...")
    pop_prey = Population.random(PREY_POP_SIZE, 8, 2)
    preys: dict[int, Prey] = {i: Prey(i, genome) for i, genome in pop_prey.genomes.items()}
    print("Done.")

    # Place organisms randomly
    sparse: dict[tuple[int, int], Organism] = {}
    for org in list(grasses.values()) + list(preys.values()):
        while True:
            r, c = np.random.randint(NROWS), np.random.randint(NCOLS)
            if (r, c) not in sparse:
                break
        org.coord = (r, c)
        sparse[(r, c)] = org
    
    # Define function to get nxn visual field
    def get_visual_field(r: int, c: int, n: int):
        is_blocked = True
        inputs = np.zeros((n, n, 3))
        for i, j in np.ndindex(n, n):
            r2, c2 = r + i - n // 2, c + j - n // 2
            r2 %= NROWS
            c2 %= NCOLS
            # Get organism at (r2, c2)
            org = sparse.get((r2, c2))
            if org is None:
                inputs[i, j, :] = GREY
                is_blocked = False
            else:
                inputs[i, j, :] = org.color
                if type(org) == Grass:
                    is_blocked = False
        return inputs, is_blocked

    # Start the game
    temperature = 100
    pygame.init()
    pygame.display.set_caption("Game 1")
    font = pygame.font.SysFont("Arial", 16)
    screen = pygame.display.set_mode((NROWS, NCOLS))
    screen.fill(GREY)
    clock = pygame.time.Clock()
    running = True

    # Draw initial state
    pixel_array = pygame.PixelArray(screen)
    for org in sparse.values():
        pixel_array[org.coord[0], org.coord[1]] = org.color

    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:                
                if event.key == pygame.K_SPACE:
                    # Save screenshot
                    pygame.image.save(screen, f"game1_{temperature:.2f}.png")
                elif event.key == pygame.K_UP:
                    temperature *= 1.1
                    print(f"Temp: {temperature:.2f}")
                elif event.key == pygame.K_DOWN:
                    temperature /= 1.1
                    print(f"Temp: {temperature:.2f}")
                elif event.key == pygame.K_c:
                    # Visualize biggest prey neural net
                    biggest: Prey = max(preys.values(), key=lambda prey: prey.genome.size())
                    biggest.genome.visualize("Biggest Prey Genome")
                    biggest.cppn.visualize("Prey CPPN with biggest genome")
                elif event.key == pygame.K_g:
                    # Visualize biggest grass neural net
                    biggest: Grass = max(grasses.values(), key=lambda grass: grass.genome.size())
                    biggest.genome.visualize("Biggest Grass Genome")
                    biggest.cppn.visualize("Grass CPPN with biggest genome")
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Get mouse position
                x, y = pygame.mouse.get_pos()
                r, c = round(x)-1, round(y)-3

                # Get organism at mouse position
                org = sparse.get((r, c))
                if org is not None:
                    # Print organism info
                    if type(org) == Grass:
                        print(f"Grass {org.id} at ({r}, {c})")
                        print(f"Energy: {org.energy:.2f}")
                        print(f"Age: {org.age}")
                        print(f"Genome size: {org.genome.size()}")
                        print(f"Phenotype size: {org.phenotype.size()}")
                        org.genome.visualize(f"Grass {org.id} at ({r}, {c})")
                        org.cppn.visualize(f"Grass CPPN")
                        # Delete organism
                        del sparse[(r, c)]
                        del grasses[org.id]
                        del pop_grass.genomes[org.id]
                        pixel_array[(r, c)] = GREY
                    elif type(org) == Prey:
                        print(f"Prey {org.id} at ({r}, {c})")
                        print(f"Energy: {org.energy:.2f}")
                        print(f"Age: {org.age}")
                        print(f"Genome size: {org.genome.size()}")
                        print(f"Phenotype size: {org.phenotype.size()}")
                        org.genome.visualize(f"Prey {org.id} at ({r}, {c})")
                        org.cppn.visualize(f"Prey CPPN")
                        # Delete organism
                        del sparse[(r, c)]
                        del preys[org.id]
                        del pop_prey.genomes[org.id]
                        pixel_array[(r, c)] = GREY
                    print()

        # Update grass
        breeding = {}
        dead = set()
        for id, grass in grasses.items():
            r, c = grass.coord

            # Update color
            if grass.color == BLUE and grass.energy > 10:
                pixel_array[grass.coord] = GREEN
                grass.color = GREEN
            elif grass.color == GREEN and grass.energy <= 10:
                pixel_array[grass.coord] = BLUE
                grass.color = BLUE

            if grass.energy > 10:
                # Get 3x3 neighborhood visual field
                inputs, is_blocked = get_visual_field(r, c, 3)
                if not is_blocked:
                    # Normalize inputs
                    inputs /= 255.0
                    inputs -= 0.5

                    # Forward propagate inputs through network
                    outputs = grass.phenotype.forward(inputs)[:, 0, 0]
                    i, j = round(outputs[0]), round(outputs[1])
                    r2, c2 = r + i, c + j
                    r2 %= NROWS
                    c2 %= NCOLS
                    breeding[id] = (r2, c2)
                    
                    # Update energy if has space
                    grass.energy += np.random.uniform(0.0, 0.1) * temperature / 100.0
            else:
                # Update energy
                grass.energy += np.random.uniform(0.0, 1) * temperature / 100.0
            
            # Handle grass death
            if grass.age > 1000 or grass.energy <= 0:
                dead.add(id)
                breeding.pop(id, None)
                del sparse[grass.coord]
                del pop_grass.genomes[id]
                pixel_array[grass.coord] = GREY
        
        # Handle grass death
        for id in dead:
            del grasses[id]
        
        # Handle grass reproduction
        for id1, id2, r, c in pair_breed(breeding, grasses, sparse):
            # Reproduce
            id3 = pop_grass.reproduce(id1, id2)
            grass3 = Grass(id3, pop_grass.genomes[id3])
            grass3.coord = (r, c)
            grasses[id3] = grass3
            sparse[grass3.coord] = grass3
            pixel_array[grass3.coord] = grass3.color

            # Penalize energy
            grasses[id1].energy -= 2.5
            grasses[id2].energy -= 2.5
        
        # Update preys
        breeding = {}
        dead = set()
        for id, prey in preys.items():
            r, c = prey.coord

            if prey.energy > 1:
                # Get 9x9 neighborhood visual field
                inputs, is_blocked = get_visual_field(r, c, 9)                
                if not is_blocked:
                    # Normalize inputs
                    inputs /= 255.0
                    inputs -= 0.5

                    # Forward propagate inputs through network
                    outputs = prey.phenotype.forward(inputs)
                    outputs = outputs[:, 0, 0]

                    # Handle movement
                    i, j = round(outputs[0]), round(outputs[1])
                    r2, c2 = r + i, c + j
                    r2 %= NROWS
                    c2 %= NCOLS
                    if (r2, c2) != prey.coord:
                        org = sparse.get((r2, c2))
                        if type(org) == Grass:
                            # Eat grass
                            grass = grasses[org.id]
                            prey.energy += grass.energy
                            del sparse[(r2, c2)]
                            del grasses[org.id]
                            del pop_grass.genomes[org.id]
                        elif type(org) == Prey:
                            # Bump into other prey
                            pass
                        elif org is None:
                            # Move prey out of its previous cell
                            prey.coord = (r2, c2)
                            del sparse[(r, c)]

                            # Place prey
                            sparse[prey.coord] = prey
                            pixel_array[(r, c)] = GREY
                            # pixel_array[prey.coord] = prey.color

                            # Use energy to move
                            prey.energy -= (0.1 * (abs(i) + abs(j)) / 2)
                
                    # Handle breeding
                    if prey.energy > 100:
                        i, j = round(outputs[2]), round(outputs[3])
                        r2, c2 = r + i, c + j
                        r2 %= NROWS
                        c2 %= NCOLS
                        breeding[id] = (r2, c2)
            
            # Update energy
            prey.energy -= np.random.uniform(0.0, 0.05) / (temperature / 100.0)

            # Update color
            prey.color = (255, 0, 255-255*np.clip((prey.energy - 10) / 100.0, 0, 1))
            pixel_array[prey.coord] = prey.color
            
            # Handle prey death
            if prey.age > 1000 or prey.energy <= 0:
                dead.add(id)
                breeding.pop(id, None)
                del sparse[prey.coord]
                del pop_prey.genomes[id]
                pixel_array[prey.coord] = GREY
        
        # Handle prey death
        for id in dead:
            del preys[id]

        # Handle prey reproduction
        for id1, id2, r, c in pair_breed(breeding, preys, sparse):
            # Reproduce
            id3 = pop_prey.reproduce(id1, id2)
            prey3 = Prey(id3, pop_prey.genomes[id3])

            # Place child prey
            prey3.coord = (r, c)
            preys[id3] = prey3
            sparse[prey3.coord] = prey3
            pixel_array[prey3.coord] = prey3.color

            # Penalize energy
            preys[id1].energy -= 25
            preys[id2].energy -= 25

        # Update temperature
        temperature *= 0.999
        if temperature < 1:
            temperature = 1

        # Update screen
        pygame.display.flip()
        clock.tick(120)
    
    pixel_array.close()
    pygame.quit()
