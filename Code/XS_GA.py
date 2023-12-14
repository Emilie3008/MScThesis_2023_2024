import matplotlib.pyplot as plt
import numpy as np 
import os
import pandas as pd
from Plausible_XS import XS, Linear_XS
from adaptative_XO_MR import AdaptativeGeneticAlgorithm

# 0. Importing the data I found online and stored locally:
# https://fispact.ukaea.uk/wiki/images/f/f0/238_HFIR-highres.txt
FLUX_LIBRARY = pd.read_csv(os.path.join(os.getcwd(),
                                         "Desktop","Mémoire", "Code_GA",
                                         "Flux_data"))

# XS is a "home-made" cross section designed to have a shape similar 
# to most of real cross sections
CROSS_SECTION = XS(FLUX_LIBRARY["Energy"].values)

# Each time a fitness will be computed, it will be stored
# in this dictionnary in order to avoid computing several times the same fitness
FITNESS = dict() 

# 1. Let's create a class that inherits from the AdaptativeGeneticAlgoritm class
#  and adapts to our application

#   1.1 Let's define some useful functions
def compute_target_rr():
    """
      Compute the reaction rate of the fine energy grid. In this application,
      the reaction rate is defined as the dot product of the flux and 
      the cross-section
    """
    # Extracting the flux from a predefined library (238-groups)
    flux = FLUX_LIBRARY['Flux'].values

    # The reaction rate is defined here as the dot product between the flux
    # and the cross-section
    reaction_rate = np.dot(flux, CROSS_SECTION)

    return reaction_rate

def weighted_average(y, energy):
    """
       Compute an average of y-values weighted by energy values. 
       weighted_avg = Σi(yi*ΔEi)/ΔE
    """
    numerator = denominator = 0
    for i in range(len(y)):
        delta_energy = energy[i+1]-energy[i]
        numerator += y[i]*delta_energy
        denominator += delta_energy

    return numerator/denominator

def get_reaction_rate(prev_cut, cut):
    """
      Compute the reaction rate on an energy group,
      which is defined by the cuts that delimit it.
    """
    energy = FLUX_LIBRARY["Energy"][prev_cut: cut+1].values
    flux = FLUX_LIBRARY["Flux"][prev_cut: cut].values

    if prev_cut == cut:
        raise ValueError("A chromosome should not contain "
                          "twice the same number")
    
    # Since energy intervals are not always the same size, 
    # the mean flux is defined as a weighted average of the flux
    # on the energy interval it is defined
    mean_flux = weighted_average(flux, energy)
    mean_xs = weighted_average(CROSS_SECTION[prev_cut:cut], energy)

    # R.R. is defined as the dot product of the flux and the cross section
    reaction_rate = mean_flux*mean_xs
    return reaction_rate


#   1.2 Let's implement our class

TARGET_RR = compute_target_rr()

class XS_GA(AdaptativeGeneticAlgorithm):
    
    def __init__(self, target_collapse,  gene_pool = range(1, 238), 
                 population_size = 100, elitism = 0.06, theta1 = 0.001, 
                 theta2 = 0.001, pm=0.5, pc=0.5):
        
        super().__init__(target_collapse-1, gene_pool, population_size,
                          elitism, theta1, theta2, pm, pc)

    class Individual():
        def __init__(self, chromosome):
            self.chromosome = sorted(chromosome)
            self.fitness = self.get_fitness()

        def get_fitness(self):
            """
              Calculate the reaction rate of the energy grid 
              defined by the individual's chromosome.
              Return the difference between the ratio of this reaction
              rate to the target reaction rate and 1, multiplied by 10e-17 
              to obtain a fitness expressed in units.
            """
            # In order to avoid computing the fitness several times for a same
            #  chromosome, it is stored in a dictionnary
            if tuple(self.chromosome) in FITNESS:
                return FITNESS[tuple(self.chromosome)]

            reaction_rate = 0
            prev_cut = 0
            for cut in self.chromosome:
                reaction_rate += get_reaction_rate(prev_cut, cut)
                prev_cut = cut
            if prev_cut < 237:
                reaction_rate += get_reaction_rate(prev_cut, 237)

            fitness = abs(reaction_rate/TARGET_RR - 1)
            FITNESS[tuple(self.chromosome)] = fitness
            return fitness

#  2. Let's find the near-optimal breakdown of energy groups
TARGET_COLLAPSE = 50
collapsing_ga = XS_GA(target_collapse=TARGET_COLLAPSE)
_, near_optimal_cut = collapsing_ga.run_adaptative_genetic_algorithm(seed=0,
                                                                tol = 0.00,
                                                                max_iter=100,
                                                                display=True)

# 3. Let's plot the data
energy = FLUX_LIBRARY['Energy'].values
flux = FLUX_LIBRARY['Flux'].values
collapsed_flux = np.zeros((TARGET_COLLAPSE, ))

i = prev_cut = 0
# Collapsing the flux according to the near optimal energy grid defined
# in the output of our algorithm
for cut in near_optimal_cut:
    collapsed_flux[i] = weighted_average(flux[prev_cut:cut], energy[prev_cut: cut+1])
    prev_cut = cut
    i+=1
if prev_cut < 236:
    collapsed_flux[i]=  weighted_average(flux[prev_cut:236], energy[prev_cut: 237])

j = i = 0
c_flux = np.zeros((len(flux),))
# The values of the collapsed flux are entered in a 
# vector whose size is equivalent to the size of the energy vector.
while i < len(flux):
    while i < near_optimal_cut[j]:
        c_flux[i] = collapsed_flux[j]
        i+=1
    j+=1
    c_flux[i] = collapsed_flux[j]

    if j == TARGET_COLLAPSE-1:
        c_flux[i:] = collapsed_flux[j]
        break

# We plot the flux as defined by the fine energy grid (238-groups) 
# and as defined by the new energy grid (TARGET_COLLAPSE-groups)
fig, ax1 = plt.subplots(figsize = (10, 5))
ax1.step(energy, flux, where = "post", color = "black", label = "238-groups")
ax1.step(energy, c_flux, where = "post", color = "olive", label = 
         "{}-groups, plausible cross section".format(TARGET_COLLAPSE))
ax1.set_yscale("log")
ax1.set_xscale("log")

# On the same graph, we plot the cross-section as a function of the energy
ax2 = ax1.twinx()
ax2.plot(energy, CROSS_SECTION, linestyle=":",color="rebeccapurple",label = "Cross-section")
ax2.set_ylabel("XS")
ax2.set_yscale("log")
ax2.spines['right'].set_color('rebeccapurple')
ax2.spines['right'].set_linewidth(2)
fig.tight_layout()
plt.title("Adaptative GA for a {}-groups collapsing with a plausible cross section".format(TARGET_COLLAPSE))
plt.legend()
plt.show()
