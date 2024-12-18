import numpy as np
import random
import matplotlib.pyplot as plt
import time

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
            # No duplicated allele allowed in a chromosome
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
            
        def __init__(self, chromosome, target):
            """
            :param chromosome: A list of integers, representing the real coded chromosome, with integers as its allele values.
            :param target: A list of integers representing the target solution.
            """
            self.chromosome = sorted(chromosome)
            self.fitness = self.get_fitness(target)
            
        def get_fitness(self, target):
            """
            :return: A numerical value being the fitness of the individual.
              If the fitness has already been computed, it is retrieved from FITNESS_LIBRARY. 
              Else, the fitness is computed and then stored in FITNESS_LIBRARY
            """
    
            return np.linalg.norm(np.array(sorted(target)) - np.array(sorted(self.chromosome)))
        
        def get_chromosome(self):
            """
            Return the sorted chromosome of the individual. 
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
        :return: a randomly generated instance of the class Individual
        """
        chromosome = [random.choice(self.allele_pool) for
                       _ in range(len(self.target))]
        return self.Individual(self.repair(chromosome), self.target)
    
    def repair(self, chromosome):
        """
        :param chromosome: A list of integer, possibly containing duplicated alleles
        :return: The same chromosome, but repaired from duplicated alleles, 
                these being replaced by random alleles from the allele pool.
        """
        registered = []
        for i in range(len(chromosome)):
            if chromosome[i] not in registered:
                registered.append(chromosome[i])
            else:
                # If the allele has already been registered, then
                # it is replaced by a random allele which is not already present in the chromosome
                chromosome[i] = self.mutated_allele(exclude = 
                                                    chromosome)
        return sorted(chromosome)
    
    def mutated_allele(self, exclude = list()):
        """
        :param exclude: (Optionnal) A list of integers
        :return: A random integer from the allele pool which is not in the exclude list
        
        """
        allele = random.choice([allele for allele in 
                                self.allele_pool if allele
                                  not in exclude])
        return allele
    
    def mutation(self, chromosome):
        """
        :param chromosome: A list of integers
        :return: The list of integers after it has undergone mutations
        """

        # Retrieving chromosome fitness before mutation
        fitness = self.Individual(chromosome, self.target).fitness
        prob = 0.5

        # If the solution is fitter than the average of the population 
        # and the mutation rate is adaptive, then the probability for 
        # it to undergo a soft-mutation is set to // self.soft_mutation //
        if fitness < self.best_fitness and self.adaptive:
            prob = self.soft_mutation

        # Iteration over all the genes of the chromosome
        for i in range(len(chromosome)):    

            # Each gene has a probability to undergo a mutation
            p = random.random()
            if p < self.mutation_rate:

                # Determining whether the mutation will be random or soft
                if random.random() > prob :
                    chromosome[i] = random.choice(self.allele_pool)
                else:
                    # The cut is shifted to the left or to the right with an equal probability
                    chromosome[i] += -1 if random.random() < 0.5 else 1

        # Chromosome repair because the mutation may have introduced duplicates
        chromosome = self.repair(chromosome)
        return chromosome
    
    def tournament(self, population):
        """
        :param population : A list of Individuals
        :return: A mating pool constructed under stochastic tournament selection
        """

        mating_pool = list()
        tournaments = {key: list() for key in range(self.NT)}

        # Every individual is randomly assigned to a tournament
        for individual in population:
            tournament = random.choice(range(self.NT))
            tournaments[tournament].append(individual)

        # Competition in every tournament
        for tournament, individuals in tournaments.items():

            # Ranking individuals inside of a tournament
            individuals = sorted(individuals, key = lambda x: x.fitness)

            for rank, individual in enumerate(individuals):
                # Fill the mating pool with a certain number of individuals depending on the weight w
                w = self.p * (1 - self.p)**rank
                w*=100
                mating_pool += [individual]* int(w)
        return mating_pool

    def create_offspring(self, parent1, parent2):
        """
        :param parent1: An instance of the class Individual
        :param parent2: An instance of the class Individual
        :return: Two instances of the class Individual, created by single-point crossover.
        """

        p = random.random()
        # There is a probability 1 - pc that no reproduction occurs.
        if p > self.pc:
            # Mutation still occurs
            p1 = self.Individual(self.mutation(parent1.get_chromosome()), self.target)
            p2 = self.Individual(self.mutation(parent2.get_chromosome()), self.target)
            return [p1,p2]
        
        # The crossover point is randomly decided
        XO = random.choice(range(1, len(self.target)-1))
        # Single-point crossover
        offspring1 = parent1.get_chromosome()[:XO] + parent2.get_chromosome()[XO:]
        offspring2 = parent2.get_chromosome()[:XO] + parent1.get_chromosome()[XO:]
        
        # Mutation of the offsprings
        mutated_offspring1 = self.Individual(self.mutation(offspring1), self.target)
        mutated_offspring2 = self.Individual(self.mutation(offspring2), self.target)

        return (mutated_offspring1, mutated_offspring2)
    
    def chose_gene_from_parents(self, parent_list, index):
        """
        :param parent_list: A list of parents, instances of the Individual class, sorted inversely according to the fitness
        :param index: An integer, index of the gene which will be transmitted to the offspring
        :return: 
        """
        p = random.random()
        probability = 0
        n = len(parent_list)

        for j in range(1, n+1):
            # Rank-based selection
            probability += 2*j/(n*(n-1))
            if probability >= p:
                return parent_list[j-1].chromosome[index]
        return parent_list[-1].chromosome[index]
    
    
    def create_offspring_from_multiple_parents(self, parent_list):
        """
        :param parent_list: A list of parents, instances of the Individual class
        :return: An offspring instance of the class Individual, created from multi-parent fitness-based reproduction
        """
        offspring = list()
        # Fitness-based reverse sorting
        parent_list = sorted(parent_list, key = lambda x: x.fitness, reverse=True)

        # Iterating over the genes of the offspring
        for i in range(len(self.target)):
            offspring.append(self.chose_gene_from_parents(parent_list, i))

        return self.Individual(self.mutation(offspring), self.target)
    
    def update_mean_fitness(self, population):
        """
        :param population: A list of instances of the Individual class
        
        Update the average fitness of the population
        """
        fitness = 0
        for indiv in population:
            fitness+= indiv.fitness
        self.mean_fitness = fitness/len(population)
    
    def run_genetic_algorithm(self, seed, max_iter = 5000, tol = 0.0, display = False):
    
        random.seed(seed)
        population = []
        generation = 0

        # 0. Random generation of the initial population
        for _ in range(self.pop_size):
            population.append(self.new_individual())

        # Fitness-based sorting of the Individuals
        population = sorted(population, key=lambda x:x.fitness)  
        
        
        while(generation < max_iter and 
            population[0].fitness > tol):
        
            if display:
                print("Best chromosome {} of fitness {}, generation {} pm {} pc {} \n".format(
                    population[0].get_chromosome(), population[0].fitness, generation, self.mutation_rate, self.pc))
            
            # A percentage // elitism // of the solutions is passed on to the new population
            new_population = population[:int(self.pop_size*self.elitism)]

            # Creation of the mating pool
            mating_pool = self.tournament(population) if not self.no_tournament else population[:self.pop_size//2]

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
                
                # Udpate of the mean fitness
                self.update_mean_fitness(population)
                # Update of the mutation rate
                self.mutation_rate = 2*(1/(1+np.exp(-(generation-self.best_generation))) - 0.5)*self.pm

        # The cost we want to minimise here is the number of generations needed to converge 
        return dict(cost=generation)
