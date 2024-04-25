
import matplotlib.pyplot as plt
from genetic_algorithm import GeneticAlgorithm
from plot_results import plot_vectors_GPT, plot_vectors_XGPT, compare_integrals
from data_extraction import coarse_energy_grid, ISOTOPE
from fitness_utils import energy_from_energy_grid
import numpy as np

for nGroups in range(7,8):
 
    genetic_algo = GeneticAlgorithm(nGroups)


    gen, fitn, energy_cut = genetic_algo.run_genetic_algorithm(seed = 0, max_iter=100) 

    compare_integrals(energy_cut)
    plot_vectors_XGPT(energy_cut)
    plot_vectors_GPT(energy_cut)

    MY_ENERGY_GRID = energy_from_energy_grid(coarse_energy_grid, energy_cut)

    plt.xscale("log")
    plt.scatter(MY_ENERGY_GRID, np.zeros(nGroups + 1), c = "red")
    plt.xlabel("Energy (MeV)")
    plt.title("({}) Energy grid ({} groups) found by comparison of the evaluated GPT "
              "and xGPT vectors (Pu239, MT = 2, 18, 102)".format(ISOTOPE, nGroups))

    plt.show()
