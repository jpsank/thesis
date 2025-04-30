from dataclasses import dataclass
from .config import *
from .genome import *


@dataclass
class Population:
    genomes: dict[int, Genome]
    species: dict[int, set[int]] = None
    next_node_id: int = 0  # Innovation number of next hidden node
    next_genome_id: int = 0  # Id of next genome

    @staticmethod
    def random(size, num_inputs, num_outputs) -> "Population":
        # Create a random population
        pop = Population(
            {i: Genome.random(num_inputs, num_outputs) for i in range(size)},
            next_node_id=num_outputs,  # Next hidden node will have this id
            next_genome_id=size  # Next genome will have this id
        )
        return pop
    
    def __post_init__(self):
        # Initialize all genomes in one species, represented by the first genome
        self.species = {0: set(self.genomes.keys())}

    def reproduce(self, i: int, j: int) -> int:
        # Crossover genomes and mutate
        if i == j:
            # Asexual reproduction
            genome = self.genomes[i].copy()
        else:
            # Sexual reproduction
            genome = crossover_genomes(self.genomes[i], self.genomes[j])
        mutate_genome(genome, self.next_node_id)

        # Add genome to population
        self.genomes[self.next_genome_id] = genome

        # Add genome to one of parents' species
        for _, members in self.species.items():
            if i in members or j in members:
                members.add(self.next_genome_id)
                break
        
        self.next_genome_id += 1
        return self.next_genome_id - 1

    def speciate(self):
        # Break up species if above threshold
        unspeciated = set()
        for repr, members in self.species.items():
            for member in members:
                if genomic_distance(self.genomes[repr], self.genomes[member]) > SPECIES_THRESHOLD:
                    members.remove(member)
                    unspeciated.add(member)

        # Add unspeciated genomes to species
        for member in unspeciated:
            for repr, members in self.species.items():
                if genomic_distance(self.genomes[repr], self.genomes[member]) < SPECIES_THRESHOLD:
                    members.add(member)
                    break
            else:
                # Create new species if no similar species found
                self.species[member] = {member}

