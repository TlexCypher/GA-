import numpy as np
import random
from typing import Callable, List, Tuple, Union

def d1_func(x: float) -> float:
    return np.sin(x) + 0.5 * np.sin(3 * x)

def d2_func(t: Tuple[int, int]) -> float:
    x, y = t
    return -x * np.sin(np.sqrt(np.abs(x))) -y * np.sin(np.sqrt(np.abs(y)))

def mutation(population: str, mutation_rate: float) -> str:
    return ''.join(
            str(1 - int(bit)) if random.random() < mutation_rate else bit
            for bit in population
    )

def binary_to_decimal(binary: str) -> int:
    return int(binary, 2)

def decimal_to_binary(decimal: int, gene_length) -> str:
    return bin(decimal)[2:].zfill(gene_length)

def gen_population(size: int) -> str:
    return ''.join([str(random.randint(0, 1)) for _ in range(size)])

def roulette_selection(populations: List[str], fitness_values: List[float]):
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

def cross_over(populations: List[str], rate: float) -> List[str]:
    new_populations = populations[:]
    for i in range(len(populations) - 1):
        for j in range(i+1, len(populations)):
            if random.random() >= rate:
                continue
            cut_point = random.randint(1, len(populations[i]) - 1)
            np1, np2 = new_populations[i][:cut_point] + new_populations[j][cut_point:], new_populations[j][:cut_point] + new_populations[i][cut_point:]
            new_populations[i], new_populations[j] = np1, np2
    return new_populations

def mutation_func(population: str, mutation_rate: float) -> str:
    return ''.join([str(1 ^ int(bit)) if random.random() < mutation_rate else bit for bit in population])

def GA(generation_num: int,
       population_num: int, 
       population_size: int, 
       fitness_func, 
       g2p,
       cross_over_rate: float,
       mutation_rate: float,
    ):
    populations = [gen_population(population_size) for _ in range(population_num)]
    fitness_values = [fitness_func(g2p(population)) for population in populations]

    for _ in range(generation_num):
        elite_index = fitness_values.index(max(fitness_values))

        next_populations = [populations[elite_index]]
        current_populations = [populations[i] for i in range(population_num) if i != elite_index]
        current_fitness_values = [fitness_values[i] for i in range(population_num) if i != elite_index]

        # Evolution.
        current_populations = roulette_selection(current_populations, current_fitness_values)
        current_populations = cross_over(current_populations, cross_over_rate) 
        current_populations = [mutation_func(population, mutation_rate) for population in current_populations]
        populations = next_populations + current_populations

        # Update fitness
        fitness_values = [fitness_func(g2p(population)) for population in populations]

    best_gene_type = populations[fitness_values.index(max(fitness_values))]
    best_pheno_type = g2p(best_gene_type)
    best_fitness_value = fitness_func(best_pheno_type)

    print("Best gene type: " + best_gene_type)
    print("Best pheno type: " + str(best_pheno_type))
    print("Best fitness value: " + str(best_fitness_value))


def b2d(binary: str) -> int:
    return int(binary, 2)

def d2b(decimal: int, gene_length=5) -> str:
    return bin(decimal)[2:].zfill(gene_length)

def gr2b(gray: str) ->  str:
    binary = ""
    for i in range(len(gray)):
        bit_sum = 0
        for j in range(i, len(gray)):
            bit_sum += int(gray[j])
            bit_sum %= 2
        binary += str(bit_sum)
    return binary

def b2gr(binary: str) -> str:
    gray = ""
    for i in range(len(binary) - 1):
        gray += str(int(binary[i]) ^ int(binary[i + 1]))
    gray += binary[-1]
    return gray

def d2vec(decimal: int) -> Tuple[int, int]:
    # To maximize exploration space, I decide scale is 8. 
    # 32 = (1, 32), (2, 16), (4, 8)
    scale = 8
    return (int(decimal // scale), int(decimal % scale))

def OneDimFuncOptimize():
    print("One Dimension Function Optimization start.")
    print("Binary Encoding Version.")
    random.seed(0)
    GA(generation_num=5,
       population_num=5,
       population_size=5,
       fitness_func=d1_func,
       g2p=lambda x: b2d(x),
       cross_over_rate=0.7,
       mutation_rate=0.3,
       )
    print("Gray Encoding Version")
    GA(generation_num=5,
       population_num=5,
       population_size=5,
       fitness_func=d1_func,
       g2p=lambda x: b2d(gr2b(x)),
       cross_over_rate=0.7,
       mutation_rate=0.3,
       )

def TwoDimFuncOptimize():
    print("Two Dimension Function Optimization start.")
    print("Binary Encoding Version.")
    random.seed(0)
    GA(generation_num=5,
       population_num=5,
       population_size=5,
       fitness_func=d2_func,
       g2p=lambda x: d2vec(b2d(x)),
       cross_over_rate=0.7,
       mutation_rate=0.3,
       )
    print("Gray Encoding Version")
    GA(generation_num=5,
       population_num=5,
       population_size=5,
       fitness_func=d2_func,
       g2p=lambda x: d2vec(b2d(gr2b(x))),
       cross_over_rate=0.7,
       mutation_rate=0.3,
       )



def main():
    OneDimFuncOptimize()
    print("---------------------------------------------------")
    TwoDimFuncOptimize()

if __name__  == '__main__':
    main()

        





