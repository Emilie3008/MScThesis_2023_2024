import numpy as np
import random


GENE_POOL = np.arange(200)
TARGET = [random.choice(GENE_POOL) for _ in range(20)]

class AdaptativeGeneticAlgorithm():
    class Individual():
        def __init__(self, chromosome):
            self.chromosome = sorted(chromosome)
            self.fitness = self.get_fitness()

        def get_fitness(self):
            """
            Fitness is defined as the norm of the difference of 
            the chromosome and the target
            """
            return np.linalg.norm(np.array(sorted(TARGET))-np.array(
                sorted(self.chromosome)))
        
    def __init__(self, target_collapse= len(TARGET), gene_pool = GENE_POOL,
                population_size = 100, elitism = 0.06, theta1 = 0.0001,
                theta2 = 0.0001, pm=0.5, pc=0.5):
        
        self.gene_pool = gene_pool
        self.population_size = population_size
        self.pm = pm
        self.pc = pc
        self.CP = self.MP = self.nc = self.nm = 0
        self.theta1 = theta1
        self.theta2 = theta2
        self.elitism = elitism
        self.target_collapse = target_collapse
        
    def mutated_gene(self, exclude = list()):
        """
          Return a random value from the gene pool which is not present in 
          the list exclude
        """
        return random.choice([gene for gene in self.gene_pool if gene not in exclude])
    
    def repair(self, chromosome):
        """
          If there are 2 of the same number in the chromosome,
          one of the doublets is replaced by a random number from the gene pool.
          Returns a list of numbers containing no redundancies.
        """
        registered = []
        for i in range(len(chromosome)):
            if chromosome[i] not in registered:
                registered.append(chromosome[i])
            else:
                chromosome[i] = self.mutated_gene(exclude = chromosome)
        return chromosome
    
    def create_individual(self):
        """
          Creates an individual with a randomly composed chromosome
        """
        chromosome = self.repair([self.mutated_gene() for _ in range(
            self.target_collapse)])
        return self.Individual(chromosome)
    
    def mate(self, parent1, parent2):
        """
         Creates two children from parent1 and parent2
         using the adaptive process.
         There is therefore a probability pc that there will be a crossover.
         For each child chromosome, there is a probability pm that 
         there is a mutation.
        """
        offspring1 = sorted(parent1.chromosome)
        offspring2 = sorted(parent2.chromosome)

        p_xo = random.random() 
        # There is a mating only if p_xo <= pc 
        if p_xo <= self.pc:
            chromosome1 = sorted(parent1.chromosome)
            chromosome2 = sorted(parent2.chromosome)

            # The crossover point is randomly chosen
            XO = random.choice(range(self.target_collapse))

            # The two children are complementary to each other.
            #  Chromosomes should be repaired if a doublet is introduced
            #  by the crossover
            offspring1 = self.repair(chromosome1[:XO] + chromosome2[XO:])
            offspring2 = self.repair(chromosome2[:XO] + chromosome1[XO:])

            # As there was a crossover, nc can be increased by 1
            self.nc += 1

            fsums =  self.Individual(offspring1).fitness  + self.Individual(
                offspring2).fitness
            fsump = parent1.fitness + parent2.fitness
            # CP measure how effective the crossover mechanism was. 
            # If CP > 0, the fitness of the children is smaller than 
            # the fitness of the parents and thus the mechanism was efficient.
            self.CP += fsump-fsums


        for offspring in [offspring1, offspring2]:
            # Each child has an independent probability of having a mutated chromosome
            p_mut = random.random()
            # There is a mating only if p_mut <= pm
            if p_mut <= self.pm:
                # Let's remember the fitness of the chromosome before it is mutated
                old_fitness = self.Individual(offspring).get_fitness()

                # The allele which will be mutated is randomly selected
                mutation = random.choice(range(self.target_collapse))
                offspring[mutation] = self.mutated_gene()

                # If the mutation has introduced a doublet, the chromosome 
                # should be repaired
                offspring = self.repair(offspring)
                new_fitness = self.Individual(offspring).get_fitness()

                # As there was a mutation, nm should be increased by 1
                self.nm += 1

                # MP measure how effective the mutation mechanism was. 
                # If MP > 0, the fitness of the chromosome is smaller after 
                # the mutation than before and thus the mechanism was efficient.
                self.MP += old_fitness-new_fitness

        return self.Individual(self.repair(offspring1)), self.Individual(
            self.repair(offspring2))
        
    def update_XO_and_MR(self):
            # Calculating the average progress value for the mutations 
            # and the cross-over
            self.CP = self.CP/self.nc if self.nc != 0 else 0
            self.MP = self.MP/self.nm if self.nm != 0 else 0

            # Adapting the rates. If the mutation mechanism is more effective
            #  than the crossover mechanism, the mutation rate should be increased 
            # and the crossover rate should be decreased. And conversely.
            if self.CP < self.MP:
                self.pc -= self.theta1
                self.pm += self.theta2
            else:
                self.pc += self.theta1
                self.pm -= self.theta2
            
            # Upper and lower bounds
            self.pc = 0.6 if self.pc > 0.6 else 0.2 if self.pc < 0.2 else self.pc
            self.pm = 0.6 if self.pm > 0.6 else 0.2 if self.pm < 0.2 else self.pm

            # Reset parameters to 0 for the next generation
            self.MP = self.CP = self.nm = self.nc = 0
    
    def run_adaptative_genetic_algorithm(self, seed, tol, max_iter, display = False):
        generation = 1
        converged = False
        population = list()
        random.seed(seed)

        # Let's create our initial population
        for _ in range(self.population_size):
            population.append(self.create_individual())
            
        while not converged:
            # Let's sort our population according to fitness
            population = sorted(population, key = lambda x:x.fitness)

            # If the fittest individual is smaller than the tolerance or 
            # the generation excedes the maximal iteration, break the loop
            if population[0].fitness <= tol or generation > max_iter:
                converged = True
                return generation, population[0].chromosome

            if display:
                print("Generation: {}\tChromosome: {}\tFitness: {}".format(
                    generation, population[0].chromosome, population[0].fitness)) 
                
            # New generation
            generation += 1
            new_generation = list()

            # Elitism = we pass the best solutions to the next generation as such
            i = int(self.elitism*self.population_size) 
            new_generation.extend(population[:i])

            # Mating
            for _ in range((self.population_size-i)//2): 
                # Parents are selected from the fittest half of the population.
                parent1 = random.choice(population[:self.population_size//2]) 
                parent2 = random.choice(population[:self.population_size//2])
                offspring1, offspring2 = self.mate(parent1, parent2) 
                new_generation.extend([offspring1, offspring2]) 

            # Let's update XO and MR thanks to the feedback collected 
            # from the previous step
            self.update_XO_and_MR()
            population = new_generation

        if display:
            print("Generation: {}\tChromosome: {}\tFitness: {}".format(
                generation, population[0].chromosome, population[0].fitness)) 
            
        return generation, population[0].chromosome

