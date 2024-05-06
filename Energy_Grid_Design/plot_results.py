import matplotlib.pyplot as plt
from data_extraction import GPT_energy_grid, XGPT_energy_grid, ISOTOPE, eigenbasis, perts
from fitness_utils import down_binning, up_binning, energy_from_energy_grid, evaluate_on_GA, extend, project_GPT_onto_eigenbasis, projection, GPT_vector, xGPT_vector, singular_matrix
import numpy as np

MT2_upper = 26 if ISOTOPE == "Pu239" else 10
MT18_upper = 55 if ISOTOPE == "Pu239" else 23
MT102_upper = 82 if ISOTOPE == "Pu239" else 41

def plot_vectors_GPT(GA_grids, method):
    evaluated_values = {"MT2":{}, "MT18":{}, "MT102":{}}

    for label in GPT_vector.keys():

        plt.xscale("log")
        plt.xlabel("E (MeV)")
        plt.ylabel("Sensitivity")
        plt.step(GPT_energy_grid, np.concatenate((GPT_vector[label], np.zeros(1))), 
                 where="post", linestyle = ":", label="200G")

        for nGroup, GA_grid in GA_grids.items():

            GPT_GA = down_binning(GPT_vector, GA_grid, GPT_energy_grid)
            GA_energy_grid = energy_from_energy_grid(GPT_energy_grid, GA_grid)
            evaluated_values[label] = np.concatenate((GPT_GA[label], np.zeros((1))))    

            plt.step(GA_energy_grid, evaluated_values[label] , where="post", label= str(nGroup) + "G")

        plt.legend()
        plt.title("200G vs {} ({}) {}".format(method, ISOTOPE, label))

        plt.show()

def plot_vectors_XGPT(GA_grid):

    GPT_GA = down_binning(GPT_vector, GA_grid, GPT_energy_grid)
    GA_energy_grid = energy_from_energy_grid(GPT_energy_grid, GA_grid)
    extended_GPT_GA = up_binning(XGPT_energy_grid, GA_energy_grid, GPT_GA)

    projected_GPT = projection(extended_GPT_GA, eigenbasis, perts)

    plt.step(np.arange(82), projected_GPT, where="pre", label="evaluated")
    plt.step(np.arange(82), xGPT_vector, where="pre", label="fine")

    plt.legend()
    plt.show()

def energy_cut_from_energy_grid(GPT_energy, energy_grid):
    """
    Retrieve the chromosomal encoding of an energy grid expressed in MeV. If the energies are not cuts of the GPT
    """
    return [np.abs(GPT_energy - energy).argmin() for energy in energy_grid][1:-1]


def compare_integrals(GA_grid):
    """
    Compare the integrals of the fine and rebinned sensitivity vectors. 
    The total integral should be conserved if the rebinning is done correctly.
    """

    nGroups = len(GA_grid) + 1
    deltaE = np.diff(GPT_energy_grid)
    for label, item in GPT_vector.items():
        evaluated_GPT = evaluate_on_GA(GPT_vector[label], GA_grid, GPT_energy_grid)
        XGPT_integral = np.sum(item*deltaE)
        GA_energy_grid =  energy_from_energy_grid(GPT_energy_grid, GA_grid)
        downbinned_integral = np.sum(evaluated_GPT*np.diff(GA_energy_grid))

        XGPT_upbinned = extend(XGPT_energy_grid, GA_energy_grid, evaluated_GPT)
        GPT_upbinned = extend(GPT_energy_grid, GA_energy_grid, evaluated_GPT)
        upbinned_GPT_integral = np.sum(GPT_upbinned*deltaE)
        upbinned_XGPT_integral = np.sum(XGPT_upbinned*np.diff(XGPT_energy_grid))
        print("{} :\n XGPT integral value = (200G) {} \n down-binned integral value = ({}G) "
              "{} \n up-binned integral value = (200G) {}\n up-binned integral value = (1500G) {}"
               " \n".format(label, XGPT_integral, nGroups, downbinned_integral, 
                            upbinned_GPT_integral, upbinned_XGPT_integral))

def plot_cumulative_cosine_GPT(GA_grid):
    GPT_GA = down_binning(GPT_vector, GA_grid, GPT_energy_grid)
    GA_energy_grid = energy_from_energy_grid(GPT_energy_grid, GA_grid)
    extended_GPT_GA = up_binning(GPT_energy_grid, GA_energy_grid, GPT_GA)
 
    for label, evaluated_values in extended_GPT_GA.items():
        values = GPT_vector[label]
        plt.title("({}) Cumulative cosine similarity for {} ({}G)".format(ISOTOPE, label, len(GA_grid)+1))
        cumulative_cosine_similarity(evaluated_values, values, GA_grid, GPT_energy_grid)

def cumulative_cosine_similarity(vector1, vector2, grid, energies):
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    prev_cut = 0
    cos_vector = []
    energy_vector = []
    for cut in grid:
        dot_product = np.dot(vector1[:cut], vector2[:cut])
        cos = dot_product/(norm_vector2*norm_vector1)
        energy_vector += list(energies[prev_cut:cut + 1])
        cos_vector += [cos]*len(energies[prev_cut:cut + 1])
            
        prev_cut = cut

    dot_product = np.dot(vector1, vector2)
    cos = dot_product/(norm_vector2*norm_vector1)
    energy_vector += list(energies[prev_cut:-1])
    cos_vector += [cos]*len(energies[prev_cut:-1])
    plt.xscale("log")
    plt.step(energy_vector, cos_vector)
    plt.show()


def plot_energy_grids(discretisations, method):
    for nGroup, energy_cut in discretisations.items():
        MY_ENERGY_GRID = energy_from_energy_grid(GPT_energy_grid, energy_cut)

        plt.xscale("log")
        plt.scatter(MY_ENERGY_GRID, nGroup * np.ones(nGroup + 1))

    plt.xlabel("Energy (MeV)")
    plt.title("Energy grids found with {} fitness function ({}, MT = 2, 18, 102)".format(method, ISOTOPE))

    plt.show()

def plot_eigenfunctions(tol):
    index = 0
    for eigen_function in eigenbasis:
        if index == 0 or index == MT2_upper or index == MT18_upper:
            title ="({}) MT2 eigenbasis (tolerance for the singular values = {})".format(ISOTOPE, tol) if index == 0 else "({}) MT18 eigenbasis (tolerance for the singular values = {})".format(ISOTOPE, tol) if index == MT2_upper else "({}) MT102 eigenbasis (tolerance for the singular values = {})".format(ISOTOPE, tol)
            plt.xscale("log")
            plt.title(title)

        if singular_matrix[index][index] > tol:
            plt.step(XGPT_energy_grid[:-1], eigen_function, label= "singular values = {}".format(singular_matrix[index][index]))
    
        index += 1
        if index == MT2_upper or index == MT18_upper or index == MT102_upper:
            # plt.legend()
            plt.show()

def plot_singular_values():
    index = 0
    while index < len(perts) - 1:
        if index == 0 or index == MT2_upper or index == MT18_upper:
            title ="({}) MT2 singular values (truncation = 1e-6)".format(ISOTOPE) if index == 0 else "{} MT18 singular values (truncation = 1e-6)".format(ISOTOPE) if index == MT2_upper else "({}) MT102 singular values (truncation = 1e-6)".format(ISOTOPE)
            plt.xscale("log")
            plt.title(title)
        
        plt.scatter(index, singular_matrix[index][index], c='black')
    
        index += 1
        if index == MT2_upper or index == MT18_upper or index == MT102_upper:
            plt.show()
