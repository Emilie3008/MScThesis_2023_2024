import numpy as np
import random
import matplotlib.pyplot as plt
import time

FITNESS_LIBRARY = {}
ALLELE_POOL = range(1500)


def generate_target(size, seed):
    """
    :param size: An integer
    :param seed: An integer
    :return: A target list, filled with //size// non-duplicated alleles from the allele pool
    """
    random.seed(seed)
    target = [random.choice(ALLELE_POOL) for _ in range(size)]
    registered = []
    for i in range(size):
        if target[i] not in registered:
            registered.append(target[i])
        else:
            target[i] = random.choice([allele for allele in ALLELE_POOL if allele not in target])

    return target


    
class GeneticAlgorithm():

    class Individual():

        def __init__(self, chromosome, target):
            """
            :param chromosome:
            :param target:
            """
            self.chromosome = sorted(chromosome)
            self.fitness = self.get_fitness(target)
            
        def get_fitness(self, target):
            """
            :return: A numerical value being the fitness of the individual.
              If the fitness has already been computed, it is retrieved from FITNESS_LIBRARY. 
              Else, the fitness is computed and then stored in FITNESS_LIBRARY
            """
            if tuple(sorted(self.chromosome)) not in FITNESS_LIBRARY:
                FITNESS_LIBRARY[tuple(self.chromosome)] = np.linalg.norm(
                    np.array(sorted(target)) - np.array(sorted(self.chromosome)))
            return FITNESS_LIBRARY[tuple(sorted(self.chromosome))]
        
        def get_chromosome(self):
            """
            :return: the sorted chromosome of the individual. 
            """
            return sorted(self.chromosome)
        
    def __init__(self, input):
        """
        :param input: An OrderedDict containing the configuration tested by irace
        """

        # Extracting usefull information from the input dictionnary
        input = input["configuration"]
        seed = input["seed"]

        # Initializing the problem
        self.target = generate_target(33, seed)
        
        # Classic parameters
        self.pop_size = 450
        self.allele_pool = ALLELE_POOL
        self.pm = input["pm"]
        self.mutation_rate = self.pm
        self.pc = input["pc"]
        self.elitism = input["elitism"]

        # Tournament selection parameters
        self.NT = 10
        self.p = input["p"]

        # Multi-parent selection parameter
        self.multi_parent = 10

        # Adaptive mutation rate parameters
        self.adaptive = True
        self.best_fitness = float('inf')
        self.soft_mutation = input["soft_mutation"]
        self.best_generation = -1 
        self.mean_fitness = -1

    
    def new_individual(self):
        """
        :return: A randomly generated Individual
        """
        chromosome = [random.choice(self.allele_pool) for
                       _ in range(len(self.target))]
        return self.Individual(self.repair(chromosome), self.target)
    
    def repair(self, chromosome):
        """
        :param chromosome: A list of integer, possibly containing duplicated alleles
        :return: The same chromosome, but repaired from duplicated alleles, 
        those being replaced with random allele from the allele pool.
        """
        registered = []
        for i in range(len(chromosome)):
            if chromosome[i] not in registered:
                registered.append(chromosome[i])
            else:
                chromosome[i] = self.mutated_allele(exclude = 
                                                    chromosome)
        return sorted(chromosome)
    
    def mutated_allele(self, exclude = list()):
        """
        :param exclude:
        """
        allele = random.choice([allele for allele in 
                                self.allele_pool if allele
                                  not in exclude])
        return allele
    
    def mutation(self, chromosome):
        """
        :param chromosome:
        :return:
        """

        fitness = self.Individual(chromosome, self.target).fitness
        prob = 0.5
        if fitness < self.best_fitness and self.adaptive:
            prob = self.soft_mutation

        for i in range(len(chromosome)):    
            p = random.random()
            if p < self.mutation_rate:
                if random.random() > prob :
                    chromosome[i] = random.choice(self.allele_pool)
                else:
                    chromosome[i] += -1 if random.random() < 0.5 else 1

        chromosome = self.repair(chromosome)
        return chromosome
    
    def tournament(self, population):
        """
        :param population:
        :return:
        """

        mating_pool = list()
        tournaments = {key: list() for key in range(self.NT)}
        for individual in population:
            tournament = random.choice(range(self.NT))
            tournaments[tournament].append(individual)

        for tournament, individuals in tournaments.items():
            individuals = sorted(individuals, key = lambda x: x.fitness)
            for rank, individual in enumerate(individuals):
                copies = self.p * (1 - self.p)**rank
                copies*=100
                mating_pool += [individual]* int(copies)
        return mating_pool
    
    def chose_gene_from_parents(self, parent_list, index):
        """
        :param parent_list:
        :param index:
        :return:
        """
        p = random.random()
        fitness = 0
        n = len(parent_list)
        for j in range(1, n+1):
            fitness += 2*j/(n*(n-1))
            if fitness >= p:
                return parent_list[j-1].chromosome[index]
        return parent_list[-1].chromosome[index]
    
    
    def create_offspring_from_multiple_parents(self, parent_list):
        offspring = list()
        parent_list = sorted(parent_list, key = lambda x: x.fitness, reverse=True)
        for i in range(len(self.target)):
            offspring.append(self.chose_gene_from_parents(parent_list, i))
        return self.Individual(self.mutation(offspring), self.target)

    def create_offspring(self, parent1, parent2):
        p = random.random()
        if p > self.pc:
            p1 = self.Individual(self.mutation(parent1.get_chromosome()), self.target)
            p2 = self.Individual(self.mutation(parent2.get_chromosome()), self.target)
            return [p1,p2]
        
        XO = random.choice(range(1, len(self.target)-1))
        offspring1 = parent1.get_chromosome()[:XO] + parent2.get_chromosome()[XO:]
        offspring2 = parent2.get_chromosome()[:XO] + parent1.get_chromosome()[XO:]
        
        mutated_offspring1 = self.Individual(self.mutation(offspring1), self.target)
        mutated_offspring2 = self.Individual(self.mutation(offspring2), self.target)
        return (mutated_offspring1, mutated_offspring2)
    
    def update_mean_fitness(self, population):
        fitness = 0
        for indiv in population:
            fitness+= indiv.fitness
        return fitness/len(population)
    
    def run_genetic_algorithm(self, seed, max_iter = 1000, tol = 0.0, display = False):
    
        random.seed(seed)
        population = []
        generation = 0

        for _ in range(self.pop_size):
            population.append(self.new_individual())

        population = sorted(population, key=lambda x:x.fitness)  
        
        
        while(generation < max_iter and 
            population[0].fitness > tol):
        
            if display:
                print("Best chromosome {} of fitness {}, generation {} pm {} pc {} \n".format(
                    population[0].get_chromosome(), population[0].fitness, generation, self.mutation_rate, self.pc))
            
            
            new_population = population[:int(self.pop_size*self.elitism)]

            mating_pool = self.tournament(population) 
            while len(new_population) < self.pop_size:
                if self.multi_parent is not None:
                    parent_list =list()
                    for _ in range(self.multi_parent):
                        parent_list.append(random.choice(mating_pool))
                    offspring = self.create_offspring_from_multiple_parents(parent_list)
                    new_population.append(offspring)

                else :
                    parent1 = random.choice(mating_pool)
                    parent2 = random.choice(mating_pool)
                    for offspring in self.create_offspring(parent1, parent2):
                        new_population.append(offspring)

            population = new_population
            generation += 1
            population = sorted(population, key=lambda x:x.fitness) 
        
            if self.adaptive : 
                if population[0].fitness < self.best_fitness:
                    self.best_fitness = population[0].fitness
                self.mean_fitness = self.update_mean_fitness(population)
                self.mutation_rate = 2*(1/(1+np.exp(-(generation-self.best_generation))) - 0.5)*self.pm
        FITNESS_LIBRARY = {}
        print(generation, "\n")
        return dict(cost=generation)