from copy import deepcopy
from gnn import Sequential
from gnn import Input, Dense, ReLu, Gurkan, Sigmoid
import numpy as np
import random

POP_SIZE = 20

class Individual:
    def __init__(self, model):
        self.model = model
        self.fitness = 0

def create_model():
    model = Sequential()
    model.add(Input(3))
    model.add(Dense(2))
    model.add(ReLu())
    model.add(Gurkan(3, gepoch=1, type='multiple', activation='ReLu'))
    model.add(Dense(2))
    model.add(Sigmoid())
    model.add(Gurkan(2, gepoch=1, type='one', activation='ReLu'))
    model.add(Dense(1))
    model.add(Sigmoid())
    model.initialize_weights()
    return model

def initialize_population():
    pop = []
    for _ in range(POP_SIZE):
        pop.append(Individual(create_model()))
    return pop

def fitness_scores(pop, X, Y):
    for ind in pop:
        ind.fitness = ind.model.forward(X, Y)[1]

def crossover(pop):
    best = int(len(pop)*0.1)
    next_gen = []
    for i in range(best):
        next_gen.append(Individual(deepcopy(pop[i].model)))
    
    while not len(next_gen) == POP_SIZE:
        model1 = deepcopy(random.choice(next_gen).model)
        model2 = deepcopy(random.choice(next_gen).model)

        new_model = Sequential()
        new_model.add(Input(3))
        new_model.add(Dense(2))
        new_model.add(ReLu())
        new_model.add(Gurkan(3, gepoch=1, type='multiple', activation='ReLu'))
        new_model.add(Dense(2))
        new_model.add(Sigmoid())
        new_model.add(Gurkan(2, gepoch=1, type='one', activation='ReLu'))
        new_model.add(Dense(1))
        new_model.add(Sigmoid())
        new_model.initialize_weights()

        for param in new_model.params:
            weight = new_model.params[param]
            for i in range(weight.shape[0]):
                for j in range(weight.shape[1]):
                    rnd = random.random()
                    if rnd < 0.47:
                        weight[i][j] = model1.params[param][i][j]
                    elif rnd < 0.94:
                        weight[i][j] = model2.params[param][i][j]
                    else:
                        weight[i][j] = random.uniform(-5, 5)

        for i, layer in enumerate(new_model.layers):
            if layer.layer_name == "Gurkan":
                rnd = random.random()
                if rnd < 0.47:
                    layer.gepoch = model1.layers[i].gepoch
                elif rnd < 0.94:
                    layer.gepoch = model2.layers[i].gepoch
                else:
                    layer.gepoch = random.randint(1, 10)
        next_gen.append(Individual(new_model))
    return next_gen

X = np.array([[0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 0, 0]]).T
Y = np.array([[0, 1, 1, 0]])

MAX_GENERATION = 10000

population = initialize_population()

for _ in range(MAX_GENERATION):
    fitness_scores(population, X, Y)

    population = sorted(population, key = lambda x:x.fitness)
    print(population[0].fitness)
    population = crossover(population)

A, loss = population[0].model.forward(X, Y)
print(A)