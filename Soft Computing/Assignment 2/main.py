# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 09:58:52 2023

@author: sonam
"""

from numpy.random import randint, rand
import numpy as np
import random

# Define the knapsack weight limit
knapsack_capacity = 35

## Randomly Initialising the list of 10 Items
n_items = 10
items = [(random.randint(1, 20), random.randint(1, 30)) for _ in range(n_items)]
print(f"Random Items List (Weight, Value) : {items}")

# The Fitness Function
def knapsack_value(x):
    total_weight = 0
    total_value = 0
    for i in range(len(x)):
        if x[i] == 1:  # If the item is selected (1), add its weight and value to the knapsack
            total_weight += items[i][0]
            total_value += items[i][1]
    # Penalize solutions that exceed the knapsack capacity
    if total_weight > knapsack_capacity:
        return 0, total_weight
    return total_value, total_weight

# tournament selection
def selection(pop, scores, k=3):
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k - 1):
        if scores[ix] > scores[selection_ix]:  # Select the solution with higher value
            selection_ix = ix
    return pop[selection_ix]

# crossover two parents to create two children
def crossover(p1, p2, r_cross):
    c1, c2 = p1.copy(), p2.copy()
    if rand() < r_cross:
        pt = randint(1, len(p1) - 2)
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]

# mutation operator
def mutation(bitstring, r_mut):
    for i in range(len(bitstring)):
        if rand() < r_mut:
            bitstring[i] = 1 - bitstring[i]

# genetic algorithm
def genetic_algorithm(objective, n_items, n_iter, n_pop, r_cross, r_mut):
    ## Randomly initialising population
    pop = [randint(0, 2, n_items).tolist() for _ in range(n_pop)]
    print(f"Initial Population: {pop}")

    ## Initialising variables for best values.
    best, best_value, best_weight = None, 0, 0
    
    ## For loop to perform genetic algorithm optimisation.
    for gen in range(n_iter):
        improved=False
        scores = []
        weights = []
        scores, weights = zip(*[objective(c) for c in pop])

        # for c in pop:
        #     score, weight = objective(c)
        #     scores.append(score)
        #     weights.append(weight)
        for i in range(n_pop):
            if scores[i] > best_value:
                best, best_value, best_weight = pop[i], scores[i], weights[i]
                print(f">{gen}, NEW BEST! Individual={best} | Value={best_value} | Weight={best_weight}")
                improved = True
        if not improved:
            print(f">{gen}, Generation saw no improvement. Best Individual={best} | Value={best_value} | Weight={best_weight}")

        selected = [selection(pop, scores) for _ in range(n_pop)]
        children = list()
        for i in range(0, n_pop, 2):
            p1, p2 = selected[i], selected[i + 1]
            for c in crossover(p1, p2, r_cross):
                mutation(c, r_mut)
                children.append(c)
        pop = children

    return [best, best_value]

# Define the total iterations, number of items, population size, crossover rate, and mutation rate
n_iter = 100
n_items = 10
n_pop = 100
r_cross = 0.9
r_mut = 1.0 / float(n_items)

# Perform the genetic algorithm search
best, value = genetic_algorithm(knapsack_value, n_items, n_iter, n_pop, r_cross, r_mut)
print('Done!')
print('Best items selected: %s' % best)
print('Total Value: %f' % value)