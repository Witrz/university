# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 10:15:37 2023

@author: TDC
"""

from sklearn import datasets
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from random import randint
import random



iris = datasets.load_iris()
X = iris.data

scaler = StandardScaler()
X = scaler.fit_transform(X)

y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

def initialization_population_mlp(size_mlp):
    pop = []
    activation = ['identity', 'logistic', 'tanh', 'relu']
    solver = ['lbfgs', 'sgd', 'adam']
    for _ in range(size_mlp):
        pop.append([random.choice(activation), random.choice(solver), randint(2, 100), randint(2, 100)])
    return pop

def crossover_mlp(mother_1, mother_2):
    child = [mother_1[0], mother_2[1], mother_1[2], mother_2[3]]
    return child

def mutation_mlp(child, prob_mut):
    for c in range(2, 4):
        if np.random.rand() < prob_mut:
            child[c] += randint(1, 10)
    return child

def function_fitness_mlp(pop, X_train, y_train, X_test, y_test):
    fitness = []
    for w in pop:
        clf = MLPClassifier(
            learning_rate_init=0.09,
            activation=w[0],
            solver=w[1],
            alpha=1e-5,
            hidden_layer_sizes=(int(w[2]), int(w[3])),
            max_iter=1000,
            n_iter_no_change=80
        )
        try:
            clf.fit(X_train, y_train)
            f = accuracy_score(clf.predict(X_test), y_test)
            fitness.append([f, clf, w])
        except Exception as e:
            print(f"Exception: {e}")
            pass
    return fitness

def ag_mlp(X_train, y_train, X_test, y_test, num_epochs=10, size_mlp=10, prob_mut=0.8):
    pop = initialization_population_mlp(size_mlp)
    fitness = function_fitness_mlp(pop, X_train, y_train, X_test, y_test)
    
    for j in range(num_epochs):
        pop_fitness_sort = sorted(fitness, key=lambda x: x[0], reverse=True)
        
        parent_1 = pop_fitness_sort[:size_mlp//2]
        parent_2 = pop_fitness_sort[size_mlp//2:]
        
        child_1 = [crossover_mlp(parent_1[i][2], parent_2[i][2]) for i in range(min(len(parent_1), len(parent_2)))]
        child_2 = [crossover_mlp(parent_2[i][2], parent_1[i][2]) for i in range(min(len(parent_1), len(parent_2)))]
        child_2 = [mutation_mlp(child, prob_mut) for child in child_2]
        
        fitness_child_1 = function_fitness_mlp(child_1, X_train, y_train, X_test, y_test)
        fitness_child_2 = function_fitness_mlp(child_2, X_train, y_train, X_test, y_test)
        
        fitness.extend(fitness_child_1)
        fitness.extend(fitness_child_2)
        fitness.sort(key=lambda x: x[0], reverse=True)
        fitness = fitness[:size_mlp]
        
    best_individual = fitness[0][1]
    return best_individual

best_mlp = ag_mlp(X_train, y_train, X_test, y_test, num_epochs=50, size_mlp=10, prob_mut=0.8)
print(best_mlp)
print("Best MLP model accuracy:", accuracy_score(best_mlp.predict(X_test), y_test))

