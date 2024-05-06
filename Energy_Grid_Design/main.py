from genetic_algorithm import GeneticAlgorithm
from plot_results import plot_vectors_GPT, plot_singular_values, plot_eigenfunctions, energy_cut_from_energy_grid, plot_cumulative_cosine_GPT
from data_extraction import ECCO33, GPT_energy_grid, main_path, ISOTOPE
import os

discretisations = {}
for ff in ["XGPT"]:
    for seed in range(80, 100):
        for nGroups in [7, 8, 10, 12, 33]:
            genetic_algo = GeneticAlgorithm(nGroups, criteria=ff)
            gen, fitn, energy_cut = genetic_algo.run_genetic_algorithm(seed = seed,
                                                                    max_iter = 100)
            with open(os.path.join(main_path,"Tests", ff, "{}_{}G_Pu_allMT.txt".format(ff, nGroups)), "+a") as file:
                file.write("{}\n".format(energy_cut))
            
#     discretisations[nGroups] = energy_cut
#     plot_cumulative_cosine_GPT(energy_cut)

# plot_vectors_GPT(discretisations, method = ff)

# plot_singular_values()
# plot_eigenfunctions(tol = 0)

# energy_cut_ecco = energy_cut_from_energy_grid(GPT_energy_grid, ECCO33)
# plot_vectors_GPT({33:energy_cut_ecco}, method="ECCO33")
