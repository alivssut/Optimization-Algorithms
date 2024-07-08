import numpy as np
import random
import matplotlib.pyplot as plt

class Genetic:
    def __init__(self, genes=None, population_size=50, chromosome_length=None, optimal_fitness=0, fitness_function=None, mutation_probability=0.5, num_parents_mating=None) -> None:
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.genes = genes
        self.optimal_fitness = optimal_fitness
        self.fitness_function = fitness_function
        self.mutation_probability = mutation_probability
        self.num_parents_mating = num_parents_mating

    def run(self, generations):
        population = self.initial_poplulation()
        fitness_scores = np.array([self.fitness_distance(self.fitness_function(chromosome.tolist()), self.optimal_fitness) for chromosome in population])
        best_fitnesses = list()

        generation_n = 0

        while generation_n < generations:
            # Evaluate fitness for each individual
            fitness_scores = np.array([self.fitness_distance(self.fitness_function(chromosome.tolist()), self.optimal_fitness) for chromosome in population])
            # print("fitness_scores: ", fitness_scores, population)

            # Select parents based on fitness scores
            parents = self.selection(x=population, weights=fitness_scores, parents_n=2)
            # print(parents)

            # Apply crossover and mutation to create offspring
            offspring = self.crossover(parents[0], parents[1])
            offspring = self.mutation(offspring)

            # Replace least fit individuals in the population
            population = self.replace_least_fit(population, np.array(fitness_scores), offspring)

            fitness_scores = np.array([self.fitness_distance(self.fitness_function(chromosome.tolist()), self.optimal_fitness) for chromosome in population])
            best_chromosome_index = fitness_scores.argsort()[0]
            best_chromosome = population[best_chromosome_index]
            best_fitness = self.fitness_function(best_chromosome.tolist())
            best_fitnesses.append(best_fitness)
            # print("fitness_scores: ", fitness_scores, population)
            if best_fitness == self.optimal_fitness:
                self.show_plot(np.arange(0, generation_n + 1), np.array(best_fitnesses))
                return best_chromosome.tolist(), fitness_scores[best_chromosome_index], generation_n
            generation_n += 1
            # print(population)

        fitness_scores = np.array([self.fitness_distance(self.fitness_function(chromosome.tolist()), self.optimal_fitness) for chromosome in population])
        best_chromosome_index = fitness_scores.argsort()[0]
        best_chromosome = population[best_chromosome_index]
        best_fitness = self.fitness_function(best_chromosome.tolist())
        self.show_plot(np.arange(0, generation_n), np.array(best_fitnesses))
        return best_chromosome.tolist(), fitness_scores[best_chromosome_index], generation_n
    
    def show_plot(self, x, y):
        plt.plot(x, y)
        plt.xlabel('generation') # X-Label
        plt.ylabel('optimal solution') # Y-Label
        plt.show()

    def initial_poplulation(self):
        population = np.chararray((self.population_size, self.chromosome_length), unicode=True)
        for i in range(self.population_size):
            population[i] = np.random.choice(self.genes, self.chromosome_length)
        return population

    def selection(self, x, weights, parents_n):
        return np.array(random.choices(x, weights=np.array(weights)/np.sum(weights), k=parents_n))

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, len(parent1) - 1)
        # print(np.core.defchararray.add(np.array(parent1[:crossover_point]), np.array(parent2[crossover_point:])))
        offspring = parent1[:crossover_point].tolist() + parent2[crossover_point:].tolist()
        return np.array(offspring)

    def mutation(self, chromosome):
        # mutation_point = random.randint(0, len(chromosome) - 1)
        mutated_chromosome = chromosome
        for i in range(0, len(mutated_chromosome)):
            if self.mutation_probability <= random.uniform(0, 1):
                mutated_chromosome[i] = np.random.choice(self.genes)
        return mutated_chromosome
    
    def replace_least_fit(self, population, fitness_scores, offspring):
        min_fitness_index = np.argmax(fitness_scores)
        # print(min_fitness_index, fitness_scores)
        population[min_fitness_index, :] = offspring
        return population
    
    def fitness_distance(self, x1, x2):
        return np.linalg.norm(x1 - x2)

def fitness_function(x):
    target =list("ali_vasisari")
    # print(x, target, sum(t!=h for t,h in zip(x, target)))
    # for t, h in zip(x, np.array(target)):
    #     print(t, h)
    return sum(t!=h for t,h in zip(x, target))

if __name__ == "__main__":
    genes = list("abcdefghijklmnopqrstuvwxyz1234567890!@#$%^&*()_+=-<>,.;:\/?{}| ")
    genetic = Genetic(genes=genes, population_size=100, fitness_function=fitness_function, chromosome_length=12, optimal_fitness=0, mutation_probability=0.825)
    print(genetic.run(generations=10000))
    