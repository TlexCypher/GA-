import random
from typing import List

class GASolver(object):
    def __init__(self, minimize: bool = False):
        self.minimize = minimize

    def roulette_selection(self, populations: List[str], fitness_values: List[float]):
        if self.minimize:
            max_fitness = max(fitness_values)
            fitness_values = [max_fitness - fv for fv in fitness_values]

        buffer = 1.0
        _fitness_values = [fv + buffer for fv in fitness_values]

        if sum(_fitness_values) == 0:
            _fitness_values = [1.0 for _ in fitness_values]

        return random.choices(
            populations,
            weights=[fv / sum(_fitness_values) for fv in _fitness_values],
            k=len(populations)
        )

    def cross_over(self, populations: List[str], rate: float) -> List[str]:
        new_populations = populations[:]
        for i in range(len(populations) - 1):
            for j in range(i + 1, len(populations)):
                if random.random() >= rate:
                    continue
                cut_point = random.randint(1, len(populations[i]) - 1)
                np1 = new_populations[i][:cut_point] + new_populations[j][cut_point:]
                np2 = new_populations[j][:cut_point] + new_populations[i][cut_point:]
                new_populations[i], new_populations[j] = np1, np2
        return new_populations

    def solve(self, generation_num: int,
              population_num: int,
              population_size: int,
              fitness_func,
              gen_population,
              g2p,
              mutation_func,
              cross_over_rate: float,
              mutation_rate: float):
        populations = [gen_population(population_size) for _ in range(population_num)]

        fitness_values = [fitness_func(g2p(population)) for population in populations]

        for _ in range(generation_num):
            if self.minimize:
                elite_index = fitness_values.index(min(fitness_values)) 
            else:
                elite_index = fitness_values.index(max(fitness_values))

            next_populations = [populations[elite_index]]
            current_populations = [populations[i] for i in range(population_num) if i != elite_index]
            current_fitness_values = [fitness_values[i] for i in range(population_num) if i != elite_index]

            current_populations = self.roulette_selection(current_populations, current_fitness_values)
            current_populations = self.cross_over(current_populations, cross_over_rate)
            current_populations = [mutation_func(population, mutation_rate) for population in current_populations]
            populations = next_populations + current_populations

            fitness_values = [fitness_func(g2p(population)) for population in populations]

        if self.minimize:
            best_index = fitness_values.index(min(fitness_values)) 
        else:
            best_index = fitness_values.index(max(fitness_values))

        best_gene_type = populations[best_index]
        best_pheno_type = g2p(best_gene_type)
        best_fitness_value = fitness_func(best_pheno_type)

        print(f"Best gene type: {best_gene_type}")
        print(f"Best pheno type: {best_pheno_type}")
        print(f"Best fitness value: {best_fitness_value}")

