import random
import numpy as np
from typing import List

class GASolver(object):
    def __init__(self):
        pass

    def gen_population(self, size: int) -> str:
        return ''.join([str(random.randint(0, 1)) for _ in range(size)])

    def roulette_selection(self, populations: List[str], fitness_values: List[float]):
        # NOTE: Add buffer of min value to ensure both sum of all elements and each element are greater than 0.
        buffer = 0.05 * (max(fitness_values) - min(fitness_values))
        _fitness_values = [fv - min(fitness_values) - buffer for fv in fitness_values]
        if sum(_fitness_values) == 0:
            _fitness_values = list(np.abs(fitness_values) + 1.0)
        return random.choices(
                populations,
                weights=[fv / sum(_fitness_values) for fv in _fitness_values],
                k=len(populations)
                )

    def cross_over(self, populations: List[str], rate: float) -> List[str]:
        new_populations = populations[:]
        for i in range(len(populations) - 1):
            for j in range(i+1, len(populations)):
                if random.random() >= rate:
                    continue
                cut_point = random.randint(1, len(populations[i]) - 1)
                np1, np2 = new_populations[i][:cut_point] + new_populations[j][cut_point:], new_populations[j][:cut_point] + new_populations[i][cut_point:]
                new_populations[i], new_populations[j] = np1, np2
        return new_populations

    def mutation_func(self, population: str, mutation_rate: float) -> str:
        return ''.join([str(1 ^ int(bit)) if random.random() < mutation_rate else bit for bit in population])

    def solve(self, generation_num: int,
           population_num: int, 
           population_size: int, 
           fitness_func, 
           g2p,
           cross_over_rate: float,
           mutation_rate: float,
        ):
        populations = [self.gen_population(population_size) for _ in range(population_num)]
        fitness_values = [fitness_func(g2p(population)) for population in populations]

        for _ in range(generation_num):
            elite_index = fitness_values.index(max(fitness_values))

            next_populations = [populations[elite_index]]
            current_populations = [populations[i] for i in range(population_num) if i != elite_index]
            current_fitness_values = [fitness_values[i] for i in range(population_num) if i != elite_index]

            # Evolution.
            current_populations = self.roulette_selection(current_populations, current_fitness_values)
            current_populations = self.cross_over(current_populations, cross_over_rate) 
            current_populations = [self.mutation_func(population, mutation_rate) for population in current_populations]
            populations = next_populations + current_populations

            # Update fitness
            fitness_values = [fitness_func(g2p(population)) for population in populations]

        best_gene_type = populations[fitness_values.index(max(fitness_values))]
        best_pheno_type = g2p(best_gene_type)
        best_fitness_value = fitness_func(best_pheno_type)

        print("Best gene type: " + best_gene_type)
        print("Best pheno type: " + str(best_pheno_type))
        print("Best fitness value: " + str(best_fitness_value))

