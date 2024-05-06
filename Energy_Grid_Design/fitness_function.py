import numpy as np
from data_extraction import GPT_energy_grid, xGPT_vector, GPT_vector, singular_matrix
from fitness_utils import cosine_similarity, down_binning, up_binning, energy_from_energy_grid, project_GPT_onto_eigenbasis

def compare_vectors_GPT_xGPT(GA_grid):
    evaluated_xGPT = project_GPT_onto_eigenbasis(GA_grid)
    return 1 - cosine_similarity(evaluated_xGPT , xGPT_vector)
 
def compare_uncertainty_GPT_xGPT(GA_grid):
    evaluated_XGPT = project_GPT_onto_eigenbasis(GA_grid)
    uncertainty_evaluated_XGPT = evaluated_XGPT @ singular_matrix @ evaluated_XGPT.T
    uncertainty_fine_xGPT = xGPT_vector @ singular_matrix @ xGPT_vector.T
    return abs(uncertainty_evaluated_XGPT - uncertainty_fine_xGPT)*1e5

def compare_vectors_GPT(GA_grid):
    GPT_GA = down_binning(GPT_vector, GA_grid, GPT_energy_grid)
    GA_energy_grid = energy_from_energy_grid(GPT_energy_grid, GA_grid)
    extended_GPT_GA = up_binning(GPT_energy_grid, GA_energy_grid, GPT_GA)

    evaluated_GPT = []
    fine_GPT = []
    for label, evaluated_values in extended_GPT_GA.items():
   
        fine_values = GPT_vector[label]
        fine_GPT.append(fine_values)
        evaluated_GPT.append(evaluated_values)

    return 1 - cosine_similarity(np.concatenate(fine_GPT), np.concatenate(evaluated_GPT))


def compare_correlation(GA_grid):
    evaluated_xGPT = project_GPT_onto_eigenbasis(GA_grid)
    return 1 - evaluated_xGPT @ singular_matrix @ xGPT_vector / (np.sqrt((evaluated_xGPT @ singular_matrix @ evaluated_xGPT )* (xGPT_vector @ singular_matrix @ xGPT_vector)))
