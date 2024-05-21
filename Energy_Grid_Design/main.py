from genetic_algorithm import GeneticAlgorithm
import os
from extract_input_data import ISOTOPE

for ff in ["uncertainty", "GPT", "XGPT"]:
    for seed in range(122, 126):
        for nGroups in [33]:
            genetic_algo = GeneticAlgorithm(nGroups, criteria=ff)
            gen, fitn, energy_cut = genetic_algo.run_genetic_algorithm(seed = seed,
                                                                    max_stagnating_iter=150,
                                                                    max_iter = 1000)
            with open(os.path.join(os.getcwd(), "Results",ISOTOPE,
                             "{}_{}G.txt".format(ff, nGroups)), "+a") as file:
                file.write("{}\n".format(energy_cut))


# dis = read_results([10, 12, 22])

# plot_grids(dis)
