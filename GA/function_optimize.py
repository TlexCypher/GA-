import numpy as np
import random
from typing import Tuple
from GA import GASolver


random.seed(0)


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


def b2d(binary: str) -> int:
    return int(binary, 2)

def d2b(decimal: int, gene_length=5) -> str:
    return bin(decimal)[2:].zfill(gene_length)

def gr2b(gray: str) ->  str:
    binary = gray[0]
    for i in range(1, len(gray)):
        binary += str(int(binary[-1]) ^ int(gray[i]))
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

def gen_population(size: int) -> str:
    return ''.join([str(random.randint(0, 1)) for _ in range(size)])

def mutation_func(population: str, mutation_rate: float) -> str:
    return ''.join([str(1 ^ int(bit)) if random.random() < mutation_rate else bit for bit in population])

def OneDimFuncOptimize():
    print("One Dimension Function Optimization start.")
    random.seed(0)
    ga_solver = GASolver(False)
    print("Binary Encoding Version.")
    ga_solver.solve(generation_num=3,
       population_num=5,
       population_size=5,
       fitness_func=d1_func,
       gen_population=lambda x:gen_population(x),
       g2p=lambda x: b2d(x),
       mutation_func=lambda x,y:mutation_func(x,y),
       cross_over_rate=0.7,
       mutation_rate=0.3,
       )
    print("####################################################")
    print("Gray Encoding Version")
    ga_solver.solve(generation_num=3,
       population_num=5,
       population_size=5,
       fitness_func=d1_func,
       gen_population=lambda x:gen_population(x),
       g2p=lambda x: b2d(gr2b(x)),
       mutation_func=lambda x,y:mutation_func(x,y),
       cross_over_rate=0.7,
       mutation_rate=0.3,
       )

def TwoDimFuncOptimize():
    random.seed(0)
    print("Two Dimension Function Optimization start.")
    ga_solver = GASolver(False)
    print("Binary Encoding Version.")
    ga_solver.solve(generation_num=3,
       population_num=5,
       population_size=5,
       fitness_func=d2_func,
       gen_population=lambda x:gen_population(x),
       g2p=lambda x: d2vec(b2d(x)),
       mutation_func=lambda x,y:mutation_func(x,y),
       cross_over_rate=0.7,
       mutation_rate=0.3,
       )
    print("Gray Encoding Version")
    ga_solver.solve(generation_num=3,
       population_num=5,
       population_size=5,
       fitness_func=d2_func,
       gen_population=lambda x:gen_population(x),
       g2p=lambda x: d2vec(b2d(gr2b(x))),
       mutation_func=lambda x,y:mutation_func(x,y),
       cross_over_rate=0.7,
       mutation_rate=0.3,
       )

def main():
    OneDimFuncOptimize()
    print("---------------------------------------------------")
    TwoDimFuncOptimize()

if __name__  == '__main__':
    main()

        





