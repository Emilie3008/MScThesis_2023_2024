from GeneticAlgorithm import GeneticAlgorithm
import numpy as np
import pandas as pd
import random
from statistics import mode
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize

TARGET_CHROMOSOME = np.array([1, 6,7, 8, 9, 22,35, 42, 58 ,63,88, 95, 101, 126, 147,156, 181, 202])
GENE_POOL = range(203)
MULTIPLE_PARENTS =  7
METHOD = "best_fitted"

class MultiParentingGA(GeneticAlgorithm):
    class Individual(GeneticAlgorithm.Individual):
        def get_fitness(self):
            return abs(np.linalg.norm(np.array(self.chromosome) - TARGET_CHROMOSOME))
        
    def __init__(self, number_of_parents, population_size=100, mutation_rate=1/len(TARGET_CHROMOSOME), gene_pool= GENE_POOL, elitism=0.05):
        self.number_of_parents = number_of_parents
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.gene_pool = gene_pool
        self.elitism = elitism
        self.target_collapse = len(TARGET_CHROMOSOME)
        
    
    def create_offspring_from_multiple_parents(self, parent_list, method):
        child_chromosome = np.zeros((len(TARGET_CHROMOSOME),))

        for i in range(len(TARGET_CHROMOSOME)):
            child_chromosome[i] = chose_gene_from_parents(parent_list, i, method)
        
        p = random.random()
        if p <= self.mutation_rate:
            mutation = random.choice(range(len(TARGET_CHROMOSOME)))
            child_chromosome[mutation] = self.mutated_gene()

        child_chromosome = self.repair(child_chromosome)
        return self.Individual(child_chromosome)
    
    def run_genetic_algorithm_multi(self, seed, tol, max_iter, display=False):
        generation = 1
        converged = False
        population = list()
        random.seed(seed)
                            
        for _ in range(self.population_size):
            chromosome = self.create_chromosome()
            population.append(self.Individual(chromosome))
            
        while not converged:
            
            population = sorted(population, key = lambda x: x.fitness)
            
            if population[0].fitness <= tol:
                converged = True
                break 
                    
            new_generation = list()
        
            i = int(self.elitism*self.population_size) 
            new_generation.extend(population[:i])

            for _ in range(self.population_size - i): 
                parents = list()
                for j in range(self.number_of_parents):
                    parents.append(random.choice(population[:self.population_size//2]))
                offspring = self.create_offspring_from_multiple_parents(parents, method=METHOD) 
                new_generation.append(offspring) 

            population = new_generation
            if display:
                print("Generation: {}\tChromosome: {}\tFitness: {}".format(generation,
                                                                           population[0].chromosome,
                                                                           population[0].fitness)) 
            generation += 1

            if generation > max_iter:
                return generation - 1, population[0].chromosome

        if display:
            print("Generation: {}\tChromosome: {}\tFitness: {}".format(generation,
                                                                       population[0].chromosome,
                                                                       population[0].fitness))
        return generation, population[0].chromosome
    
    def run_genetic_algorithm(self, seed, tol, max_iter, display = False):
        generation = 1
        converged = False
        population = list()
        random.seed(seed)
                            
        for _ in range(self.population_size):
            chromosome = self.create_chromosome()
            population.append(self.Individual(chromosome))
            
        while not converged:
            
            population = sorted(population, key = lambda x:x.fitness)
            
            
            if population[0].fitness <= tol:
                converged = True
                break 
                    
            new_generation = list()
        
            i = int(self.elitism*self.population_size) 
            new_generation.extend(population[:i])

            for _ in range(self.population_size - i): 
                parent1 = random.choice(population[:self.population_size//2]) 
                parent2 = random.choice(population[:self.population_size//2])
                offspring = self.create_offspring(parent1, parent2) 
                new_generation.append(offspring) 

            population = new_generation
            if display:
                print("Generation: {}\tChromosome: {}\tFitness: {}".format(generation,
                                                                           population[0].chromosome,
                                                                           population[0].fitness)) 
            generation += 1

            if generation > max_iter:
                return generation - 1, population[0].chromosome

        if display:
            print("Generation: {}\tChromosome: {}\tFitness: {}".format(generation,
                                                                       population[0].chromosome,
                                                                       population[0].fitness)) 
        return generation, population[0].chromosome
        
    
def chose_gene_from_parents(parent_list, i, method):
    if method == "best_fitted":
        p = random.random()
        fitness = 0
        n = len(parent_list)
        parent_list = sorted(parent_list, key = lambda x: x.fitness, reverse=True)
        for j in range(1, n+1):
            fitness += j/(n*(n-1))
            if fitness >= p:
                return parent_list[j-1].chromosome[i]
        return parent_list[-1].chromosome[i]

    elif method == "uniform":
        parent_genes = []
        for parent in parent_list:
            parent_genes.append(parent.chromosome[i])
        return random.choice(parent_genes)
    
    elif method == "most_common":
        parent_genes = []
        for parent in parent_list:
            parent_genes.append(parent.chromosome[i])
        return mode(parent_genes)
    
    else:
        raise ValueError('Method {} not recognized. Please chose '
                         'a selection method among the following keywords: '
                         'best_fitted, uniform, most_common'.format(method))
    

results = pd.read_csv(os.path.join(os.getcwd(), "Desktop", "Mémoire", "Code_GA", "Quick analysis"))

def objective_function(params, x, y):
    a, b, c = params
    y_pred = a * x**2 + b * x + c
    return np.mean((y - y_pred)**2)


for pop, opt in [(80, 9), (100,11), (120,11)]:
    res = results[results["Population"] == pop]
    data_opt = res[res["Parents"] == opt]
    data_2 = res[res["Parents"] == 2]
    plt.scatter(data_opt["Parents"], data_opt["Iterations"], label = "Iterations")
    plt.scatter(data_2["Parents"], data_2["Iterations"], label = "Iterations")
    plt.xlabel("Number of parents")
    plt.ylabel("Iterations before convergence")
    plt.title("Needed iterations to converge : {} vs 2 parents. Population size = {}".format(opt, pop))
    plt.show()

    plt.scatter(data_opt["Parents"], data_opt["Execution time"], label = "Execution time")
    plt.scatter(data_2["Parents"], data_2["Execution time"], label = "Execution time")
    plt.xlabel("Number of parents")
    plt.ylabel("Execution time (s)")
    plt.title("Comparison of the execution time : {} vs 2 parents. Population size = {}".format(opt, pop))
    plt.show()

