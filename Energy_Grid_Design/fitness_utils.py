import numpy as np
from data_extraction import XGPT_energy_grid, GPT_energy_grid, GPT_vector, xGPT_vector, perts, eigenbasis, singular_matrix 


def down_binning(GPT_sensitivities, GA_energy_grid, GPT_energy_grid):
    downbinned = {}
    for label, sensitivities in GPT_sensitivities.items():
        downbinned[label] = evaluate_on_GA(sensitivities, GA_energy_grid, GPT_energy_grid)
    return downbinned

def evaluate_on_GA(GPT_sensitivities, GA_energy_grid, GPT_energy_grid):
    sensitivities_evaluated = np.zeros((len(GA_energy_grid) + 1,))
    j = 0
    energies = []
    for i in range(len(GPT_sensitivities)):

        cut = len(GPT_sensitivities) if j >= len(GA_energy_grid) else GA_energy_grid[j]
        prev_cut = 0 if j == 0 else GA_energy_grid[j - 1]

        if i >= prev_cut and i < cut:
            sensitivities_evaluated[j] += GPT_sensitivities[i]*(GPT_energy_grid[i+1]-GPT_energy_grid[i])
            energies.append(GPT_energy_grid[i+1] - GPT_energy_grid[i])

        else :
            sensitivities_evaluated[j] /= np.sum(energies) 

            j += 1
            energies = []
            cut = len(GPT_sensitivities) if j >= len(GA_energy_grid) else GA_energy_grid[j]
            prev_cut = 0 if j == 0 else GA_energy_grid[j - 1]
            sensitivities_evaluated[j] += GPT_sensitivities[i]*(GPT_energy_grid[i+1]-GPT_energy_grid[i])

            energies.append(GPT_energy_grid[i+1] - GPT_energy_grid[i])

    sensitivities_evaluated[-1] /= np.sum(energies) 
    return sensitivities_evaluated

# Extracting the energy discretisation defined by the GA grid
def energy_from_energy_grid(energy_discretisation, GA_grid):
    coarse_energy = np.zeros((len(GA_grid) + 2))
    coarse_energy[0] = energy_discretisation[0]
    i = 1
    for cut in GA_grid:
        coarse_energy[i] = energy_discretisation[cut]
        i += 1
    coarse_energy[-1] = energy_discretisation[-1]

    return coarse_energy

# Upbinning to 1500G
def up_binning(XGPT_energy_grid, GPT_energy_grid, down_binned_vector):
    upbinned = {}
    for label, sens in down_binned_vector.items():
        upbinned[label] = extend(XGPT_energy_grid, GPT_energy_grid, sens)
    return upbinned

def extend(XGPT_energy_grid, GPT_energy_grid, down_binned_vector):
    up_binned_vector = np.zeros( (len(XGPT_energy_grid) - 1, ) )
    j = 1
    for i in range(1, len(XGPT_energy_grid)):
        prev_energy = GPT_energy_grid[j - 1]
        energy = GPT_energy_grid[j]

        if XGPT_energy_grid[i] >= prev_energy and XGPT_energy_grid[i] <= energy:
            up_binned_vector[i - 1] = down_binned_vector[j - 1]
            if XGPT_energy_grid[i] == energy:
                j += 1
        else:
            SL = down_binned_vector[j - 1]
            SR = down_binned_vector[j]
            DEL = (energy - XGPT_energy_grid[i - 1])
            DER = (XGPT_energy_grid[i] - energy)
            up_binned_vector[i-1] = (SL * DEL + SR * DER)/(DEL + DER)
            j += 1

    return up_binned_vector

# Projection onto the eigen basis 
def projection(dict_vector, eigenbasis, labels, energy_grid = XGPT_energy_grid):
    projected_vector = []
    delta_E = np.diff(energy_grid)
    for label, index in labels.items():
        if index == 0:
            continue
        _, label, _ = label.split("_")
        vector = dict_vector[label]

        projected_vector.append(np.sum(np.array(vector)*np.array(eigenbasis[index-1])))
        
    return np.array(projected_vector)

def project_GPT_onto_eigenbasis(GA_grid):
    GPT_GA = down_binning(GPT_vector, GA_grid, GPT_energy_grid)
    GA_energy_grid = energy_from_energy_grid(GPT_energy_grid, GA_grid)
    extended_GPT_GA = up_binning(XGPT_energy_grid, GA_energy_grid, GPT_GA)
    projected_GPT = projection(extended_GPT_GA, eigenbasis, perts)
    return projected_GPT


# Comparing xGPT et evaluated GPT
def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity

