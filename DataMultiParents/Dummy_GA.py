from GeneticAlgorithm import GeneticAlgorithm
import matplotlib.pyplot as plt
import numpy as np 
import os
import pandas as pd

FLUX_LIBRARY = pd.read_csv(os.path.join(os.getcwd(), "Flux_data"))

def weighted_average(y, energy):
    if len(energy)==1:
        return y[0]
    numerator = denominator = 0
    for i in range(len(y)):
        delta_energy = energy[i+1]-energy[i]
        numerator += y[i]*delta_energy
        denominator += delta_energy
    if denominator == 0:
        return np.mean(y)
    return numerator/denominator

def cross_section(energy):
    return np.array([600200 for i in range(len(energy))])

def get_reaction_rate(prev_cut, cut):
    if prev_cut == cut:
        return 0
    energy = FLUX_LIBRARY["Energy"][prev_cut: cut+1]
    flux = FLUX_LIBRARY["Flux"][prev_cut: cut]
    mean_flux = weighted_average(flux.values, energy.values)
    reaction_rate = mean_flux*np.mean(cross_section(energy))
    return reaction_rate

class Dummy_GA(GeneticAlgorithm):
    class Individual(GeneticAlgorithm.Individual):
        def get_fitness(self):
            reaction_rate = 0
            prev_cut = 0
            for cut in sorted(self.chromosome):
                reaction_rate += get_reaction_rate(prev_cut, cut)
                prev_cut = cut
            reaction_rate += get_reaction_rate(prev_cut, len(FLUX_LIBRARY['Energy'].values)-1)
            return reaction_rate*10e-21
          
    def target_rr(self):
        energy = self.flux_library['Energy'].values
        flux = self.flux_library['Flux'].values
        reaction_rate = np.dot(flux, cross_section(energy))
        return reaction_rate*10e-21
    

TARGET_COLLAPSE = 50
# collapsing_ga = Dummy_GA(TARGET_COLLAPSE, flux_library=FLUX_LIBRARY)
# _, optimal_cut = collapsing_ga.run_genetic_algorithm(seed=1, tol = 0.1, max_iter=100, display=True)


optimal_cut = [10, 11, 13, 14, 15, 21, 22, 33, 38, 39, 40, 41, 43, 46, 49, 52, 56, 57, 59, 60, 62, 64, 68, 73, 83, 90, 93, 100, 128, 136, 146, 176, 206, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 228]
energy = FLUX_LIBRARY['Energy'].values
flux = FLUX_LIBRARY['Flux'].values

collapsed_flux = np.zeros((TARGET_COLLAPSE, ))
i = prev_cut = 0

for cut in optimal_cut:
    collapsed_flux[i] = weighted_average(flux[prev_cut:cut], energy[prev_cut: cut+1])
    prev_cut = cut
    i+=1

collapsed_flux[i] = weighted_average(flux[prev_cut:len(flux)-2], energy[prev_cut:])

j = i = 0

c_flux = np.zeros((len(flux),))
while i < len(flux):
    while i < optimal_cut[j]:
        c_flux[i] = collapsed_flux[j]
        i+=1
    j+=1
    c_flux[i] = collapsed_flux[j]

    if j == TARGET_COLLAPSE-1:
        c_flux[i:] = collapsed_flux[j]
        break


fig, ax1 = plt.subplots(figsize = (10, 5))


ax1.step(energy, flux, where = "post", color = "black", label = "238-groups")
ax1.step(energy, c_flux, where = "post", color = "red", label = "50-groups")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("Energy (eV)"),
ax1.set_ylabel("Flux (barn s-1)")
ax1.set_title("Discreprancy for a 50-groups collapsing with a constant cross-section")
plt.legend()


ax2 = ax1.twinx()
ax2.step(energy[5:], (abs(flux[5:])-abs(c_flux[5:]))/flux[5:],":", where = "post", color = "lightpink", label = "discreprancy")
ax2.set_ylabel("Discreprancy (%)")
fig.tight_layout()

plt.show()