from GA import GASolver
import random
from typing import List
from math import sqrt
import numpy as np


class TSP(object):
    def __init__(self, city_count: int):
        self.city_count = city_count
        self.cities = np.array([
            [0, 1], 
            [6, 7],
            [3, 8],
            [1, 4],
            [3, 3]]
)

    def gen_population(self, size: int) -> List[int]:
        return self.path2coding(list(np.random.permutation(size)))

    def mutation_func(self, path: List[int], mutation_rate: float)-> List[int]:
        for i in range(len(path)):
            if random.random() < mutation_rate:
                candidates = [j for j in range(len(path) - i) if j != path[i]]
                if len(candidates) == 0:
                    continue
                path[i] = random.choice(candidates)
        return path

    def path2coding(self, path: List[int]) -> List[int]:
        coding, orders = [], list(range(len(self.cities)))
        for p in path:
            coding.append(orders.index(p))
            orders.remove(p)
        return coding


    def coding2path(self, coding: List[int]) -> List[int]:
        path, orders = [], list(range(len(self.cities)))
        for code in coding:
            path.append(orders[code])
            orders.remove(orders[code])
        return path

    def dist(self, path: List[int]):
        total_distance = 0.0
        for i in range(len(path) - 1):
            total_distance += np.linalg.norm(self.cities[path[i + 1]] - self.cities[path[i]]) 
        total_distance += np.linalg.norm(self.cities[path[-1]]- self.cities[path[0]])
        return total_distance

def main():
    random.seed(0)
    np.random.seed(0)
    print("Start solving TSP with Genetic Algorithm.")
    tsp = TSP(5)
    ga_solver = GASolver(minimize=True)
    ga_solver.solve(
       generation_num=2000,
       population_num=5,
       population_size=5,
       fitness_func=lambda x:tsp.dist(x),
       gen_population=lambda x:tsp.gen_population(x),
       g2p=lambda x: tsp.coding2path(x),
       mutation_func=lambda x,y:tsp.mutation_func(x,y),
       cross_over_rate=0.7,
       mutation_rate=0.05,
       )

if __name__ == '__main__':
    main()
