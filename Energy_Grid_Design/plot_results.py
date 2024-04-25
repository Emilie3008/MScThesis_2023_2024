import matplotlib.pyplot as plt
from data_extraction import coarse_energy_grid, fine_energy_grid
from fitness_utils import down_binning, energy_from_energy_grid, evaluate_on_GA, extend, get_GPT_on_eigenbasis, GPT_vector, xGPT_vector
import numpy as np

def plot_vectors_GPT(GA_grid):
    GPT_GA = down_binning(GPT_vector, GA_grid, coarse_energy_grid)
    GA_energy_grid = energy_from_energy_grid(coarse_energy_grid, GA_grid)
    energy = coarse_energy_grid
    for label, evaluated_values in GPT_GA.items():

        evaluated_values = np.concatenate((evaluated_values, np.zeros((1))))
        fine_values = np.concatenate((GPT_vector[label], np.zeros(1)))

        plt.xscale("log")
        
        plt.step(energy, fine_values, where="post", label="fine")
        plt.step(GA_energy_grid, evaluated_values , where="post", label="evaluated")

        plt.legend()
        plt.title(label)
        plt.show()

def plot_vectors_XGPT(GA_grid):
    projected_GPT = get_GPT_on_eigenbasis(GA_grid)

    plt.step(np.arange(82), (projected_GPT - 
                             np.mean(projected_GPT))/np.std(projected_GPT), where="pre", label="evaluated")
    plt.step(np.arange(82),( xGPT_vector - 
                            np.mean(xGPT_vector))/np.std(xGPT_vector), where="pre", label="fine")

    plt.legend()
    plt.show()

def energy_cut_from_energy_grid(coarse_energy, energy_grid):
    return [np.abs(coarse_energy - energy).argmin() for energy in energy_grid][1:-1]


def compare_integrals(GA_grid):

    nGroups = len(GA_grid) + 1

    for label, item in GPT_vector.items():
        evaluated_GPT = evaluate_on_GA(GPT_vector[label], GA_grid, coarse_energy_grid)
        fine_integral = np.trapz(item, coarse_energy_grid[:-1])
        GA_energy_grid =  energy_from_energy_grid(coarse_energy_grid, GA_grid)
        downbinned_integral = np.trapz(evaluated_GPT, GA_energy_grid[:-1])

        fine_upbinned = extend(fine_energy_grid, GA_energy_grid, evaluated_GPT)
        coarse_upbinned = extend(coarse_energy_grid, GA_energy_grid, evaluated_GPT)
        upbinned_coarse_integral = np.trapz(coarse_upbinned, coarse_energy_grid[:-1])
        upbinned_fine_integral = np.trapz(fine_upbinned , fine_energy_grid[:-1])
        print("{} :\n fine integral value = (200G) {} \n down-binned integral value = ({}G) "
              "{} \n up-binned integral value = (200G) {}\n up-binned integral value = (1500G) {}"
               " \n".format(label, fine_integral, nGroups, downbinned_integral, 
                            upbinned_coarse_integral, upbinned_fine_integral))


