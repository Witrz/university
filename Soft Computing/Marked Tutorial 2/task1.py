import random

# Define the constants
POPULATION_SIZE = 100  # Size of the population
MUTATION_RATE = 0.1   # Probability of mutation
GENERATIONS = 10      # Number of generations

# Define the input values
inputs = [4, -2, 7]

# Define the fitness function (maximize Y)
def fitness_function(weights):
    y = sum(w * x for w, x in zip(weights, inputs))
    return y

# Initialize the population with random weights
def initialize_population():
    population = []
    for _ in range(POPULATION_SIZE):
        weights = [random.uniform(-10, 10) for _ in range(3)]  # Initialize weights randomly
        population.append(weights)
    return population

# Select parents for mating using tournament selection
def select_parents(population):
    parents = []
    for _ in range(POPULATION_SIZE):
        tournament = random.sample(population, 5)  # Select 5 random individuals
        tournament.sort(key=lambda ind: -fitness_function(ind))  # Sort by fitness (maximize)
        selected_parent = tournament[0]
        parents.append(selected_parent)
    return parents

# Perform crossover to create offspring
def crossover(parent1, parent2):
    crossover_point = random.randint(1, 2)  # Randomly choose crossover point
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Perform mutation on an individual
def mutate(individual):
    for i in range(3):
        if random.random() < MUTATION_RATE:
            individual[i] += random.uniform(-1, 1)  # Add a small random value

# Main GA loop
def genetic_algorithm():
    population = initialize_population()
    
    for generation in range(GENERATIONS):
        parents = select_parents(population)
        new_population = []

        for i in range(0, POPULATION_SIZE, 2):
            parent1, parent2 = parents[i], parents[i + 1]
            child1, child2 = crossover(parent1, parent2)
            mutate(child1)
            mutate(child2)
            new_population.extend([child1, child2])
        
        population = new_population

        best_individual = max(population, key=fitness_function)
        best_fitness = fitness_function(best_individual)
        
        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}, Best Weights = {best_individual}")

    best_individual = max(population, key=fitness_function)
    best_fitness = fitness_function(best_individual)
    
    print("\nFinal Solution:")
    print(f"Best Fitness = {best_fitness}")
    print(f"Best Weights = {best_individual}")

if __name__ == "__main__":
    genetic_algorithm()