import numpy as np
from GA_irace_format import GeneticAlgorithm
from irace import irace

parameters_table = '''
pm "--mutation_rate" r (0.04, 0.05)
soft_mutation "--soft_mutation" r (0.8, 1)
pc "--crossover_rate" r (0.8, 1) 
p "--parameter p" r (0.3, 0.45)
elitism "--elitism" r (0.02, 0.05)
'''

instances = np.arange(10)

scenario = dict(
    instances = instances,
    maxExperiments = 1000,
    debugLevel = 3,
    digits = 5,
    parallel= 4, 
    logFile = "")

def target_runner(experiment, scenario):
    GA = GeneticAlgorithm(experiment)
    seed = experiment["seed"]
    dict_ = GA.run_genetic_algorithm(seed)
    return dict_



tuner = irace(scenario, parameters_table, target_runner)    
best_confs = tuner.run()
# Pandas DataFrame
print(best_confs)