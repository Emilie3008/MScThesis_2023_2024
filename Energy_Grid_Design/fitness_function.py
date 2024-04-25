import numpy as np
from data_extraction import coarse_energy_grid
from fitness_utils import cosine_similarity, down_binning, up_binning, energy_from_energy_grid, get_GPT_on_eigenbasis, xGPT_vector, GPT_vector

def compare_vectors_GPT_xGPT(GA_grid):
    projected_GPT = get_GPT_on_eigenbasis(GA_grid)
    projected_GPT = (projected_GPT - np.mean(projected_GPT))/np.std(projected_GPT)
    XGPT_vector = (xGPT_vector - np.mean(xGPT_vector))/np.std(xGPT_vector)
    return 1 - cosine_similarity(projected_GPT , XGPT_vector)
 
def compare_uncertainty_GPT_xGPT(GA_grid):
    projected_GPT = get_GPT_on_eigenbasis(GA_grid)
    uncertainty_GPT = projected_GPT.T @ projected_GPT
    uncertainty_xGPT = xGPT_vector.T @ xGPT_vector
    return abs(uncertainty_GPT-uncertainty_xGPT)

def compare_vectors_GPT(GA_grid):
    GPT_GA = down_binning(GPT_vector, GA_grid, coarse_energy_grid)
    GA_energy_grid = energy_from_energy_grid(coarse_energy_grid, GA_grid)
    extended_GPT_GA = up_binning(coarse_energy_grid, GA_energy_grid, GPT_GA)
    similarities = 0
    for label, evaluated_values in extended_GPT_GA.items():
        values = GPT_vector[label]
        similarities += cosine_similarity(evaluated_values, values)

    return 1 - similarities/len(extended_GPT_GA)

