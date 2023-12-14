import numpy as np
import random

class GeneticAlgorithm():
    class Individual():
        def __init__(self, chromosome):
            self.chromosome = sorted(chromosome)
            self.fitness = self.get_fitness()
            
        def get_fitness(self):
            raise NotImplemented

            
    def __init__(self, target_collapse, population_size = 100, mutation_rate= 0.1, gene_pool=range(1, 238),
                 elitism=0.1, flux_library = None):
        
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.target_collapse = target_collapse - 1
        self.gene_pool= gene_pool
        self.elitism = elitism
        self.flux_library = flux_library
        self.target_reaction_rate = self.target_rr() 

    def mutated_gene(self, exclude = list()):
        gene = random.choice([gene for gene in self.gene_pool if gene not in exclude])
        return gene
    
    def create_chromosome(self):
        chromosome = [self.mutated_gene() for _ in range(self.target_collapse)]
        return self.repair(chromosome)

    def create_offspring(self, parent1, parent2):
        """
        Mate the current object (parent1) with a second object (parent2) and create a new offspring,
        whose genes are drawn from the 2 parents and with a few mutations
        """ 
        chromosome1 = parent1.chromosome
        chromosome2 = parent2.chromosome
        XO = random.choice(range(len(chromosome1)))
        offspring = chromosome1[:XO] + chromosome2[XO:]
        
        for _ in range(int(self.mutation_rate*self.target_collapse)):
            mutation = random.choice(range(len(chromosome1)))
            offspring[mutation] = self.mutated_gene()

        offspring = self.repair(offspring)
        return self.Individual(offspring)

    def repair(self, chromosome):
        registered = []
        for i in range(len(chromosome)):
            if chromosome[i] not in registered:
                registered.append(chromosome[i])
            else:
                chromosome[i] = self.mutated_gene(exclude = chromosome)
                
        return chromosome
        
    def run_genetic_algorithm(self, seed, tol, max_iter, display = False, variable_mutation_rate = False):
        generation = 1
        converged = False
        population = list()
        random.seed(seed)
        
        previous_fitness = np.nan
                            
        for _ in range(self.population_size):
            chromosome = self.create_chromosome()
            population.append(self.Individual(chromosome))
            
        while not converged:
            
            population = sorted(population, key = lambda x:abs(x.fitness/self.target_reaction_rate - 1))
            
            
            if abs(population[0].fitness/self.target_reaction_rate - 1) <= tol:
                converged = True
                break 
                
            if variable_mutation_rate:
                if previous_fitness == population[0].fitness:
                    prev += 1
                    if prev == 10 and abs(population[0].fitness/self.target_reaction_rate -1) <= tol*2:
                        self.mutation_rate /= 2
                    if prev == 60:
                        self.mutation_rate = 0.1
                else:
                    prev = 0
                    previous_fitness = population[0].fitness
                    
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
                print("Generation: {}\tChromosome: {}\tFitness: {}\tTarget: {}\t Difference: {}".format(generation,
                                                                                                    population[0].chromosome,
                                                                                                    population[0].fitness, 
                                                                                                    self.target_reaction_rate, 
                                                                                                    abs(population[0].fitness/self.target_reaction_rate -1))) 
            generation += 1

            if generation > max_iter:
                return generation - 1, population[0].chromosome

        if display:
            print("Generation: {}\tChromosome: {}\tFitness: {}\tTarget: {}\t Difference: {}".format(generation,
                                                                                                    population[0].chromosome,
                                                                                                    population[0].fitness, 
                                                                                                    self.target_reaction_rate, 
                                                                                                    abs(population[0].fitness/self.target_reaction_rate - 1))) 
        return generation, population[0].chromosome