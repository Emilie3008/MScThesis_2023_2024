from GeneticAlgorithm import GeneticAlgorithm
import numpy as np
import pandas as pd
import random
import os
from statistics import mode

GENE_POOL = range(400)
MULTIPLE_PARENTS = 6
METHOD = "best_fitted"
TARGET = [1, 3, 4, 7, 8, 23, 32, 41, 45, 49, 51 , 56, 63, 72, 81, 88, 91, 92, 98, 100, 110, 113, 142]

class MultiParentingGA(GeneticAlgorithm):
    class Individual(GeneticAlgorithm.Individual):
        def get_fitness(self):
            return np.linalg.norm(np.array(self.chromosome) - TARGET)
        
    def __init__(self, number_of_parents, population_size=100, mutation_rate=0.1, gene_pool= GENE_POOL, elitism=0.1):
        self.number_of_parents = number_of_parents
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.gene_pool = gene_pool
        self.elitism = elitism
        self.target_collapse = len(TARGET)
        
    
    def create_offspring_from_multiple_parents(self, parent_list, method):
        child_chromosome = np.zeros((len(TARGET),))

        for i in range(len(TARGET)):
            child_chromosome[i] = chose_gene_from_parents(parent_list, i, method)
            
        for _ in range(int(self.mutation_rate*self.target_collapse)):
            mutation = random.choice(range(len(TARGET)))
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
                break

            yield generation, population[0].fitness

        if display:
            print("Generation: {}\tChromosome: {}\tFitness: {}".format(generation,
                                                                       population[0].chromosome,
                                                                       population[0].fitness))
        yield generation, population[0].fitness
        
    
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


multi_parenting_GA = MultiParentingGA(number_of_parents=MULTIPLE_PARENTS)

conv = pd.DataFrame()
for it, fit in multi_parenting_GA.run_genetic_algorithm_multi(seed = 1, tol = 0, max_iter= 20, display=True):
    new_line = pd.DataFrame(data = [it, fit], index = ["Iteration", "Fitness"]).T
    conv = pd.concat([conv, new_line])

conv.to_csv(os.path.join(os.getcwd(), "Desktop", "Mémoire", "Code_GA", "BF_"+ str(MULTIPLE_PARENTS)), index = False)