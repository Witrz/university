
genes = 2
chromosomes = 10
lb = -5
ub = 5
populationSize = (chromosomes, genes)
generations = 50
mattingPoolSize = 6
offSpringSize = populationSize - mattingPoolSize

import numpy
population = numpy.random.uniform(lb, ub, populationSize)
fitness = numpy.sum(population*population, axis=1)

# Following statement will create an empty two dimensional array to storeparents
parents = numpy.empty((mattingPoolSize, population.shape[1]))

# A loop to extract one parent in each iteration
for p in range(mattingPoolSize):
    # Finding index of fittest chromosome in the population
    fittestIndex = numpy.where(fitness == numpy.max(fitness))
    # Extracting index of fittest chromosome
    fittestIndex = fittestIndex[0][0]
    # Copying fittest chromosome into parents array
    parents[p, :] = population[fittestIndex, :]
    # Changing fitness of fittest chromosome to avoid reselection of that chromosome
    fitness[fittestIndex] = -1

offspring = numpy.empty((offSpringSize, population.shape[1]))

for k in range(offSpringSize):
    crossoverPoint = numpy.random.randint(0, genes)

    parent1Index = k%parents.shape[0]
    parent2Index = (k+1)%parents.shape[0]

    offspring[k, 0: crossoverPoint] = parents[parent1Index, 0: crossoverPoint]  
    
    offspring[k, crossoverPoint:] = parents[parent2Index, crossoverPoint:]
