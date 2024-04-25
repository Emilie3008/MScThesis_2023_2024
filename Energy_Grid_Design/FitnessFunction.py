import serpentTools
import numpy as np
import os
from scipy.interpolate import interp1d
from numpy import trapz

# -------------------------------------- EXTRACTING THE DATA AT THE BEGINNING OF THE RUN  -------------------------------------- 

fine_energy_grid = np.zeros((1501,))
coarse_energy_grid = np.zeros((201,))

with open(os.path.join(os.getcwd(), 
                       "Desktop","Mémoire", 'MScThesis_2023_2024',
                       "Code", "xGPT&GPT",
                        "1500G.txt"), "r") as file:
    i = 0
    for line in file:
        energy, _= line.split("\n")
        fine_energy_grid[i] = float(energy)
        i += 1


with open(os.path.join(os.getcwd(),
                        "Desktop","Mémoire", 'MScThesis_2023_2024',
                       "Code", "xGPT&GPT",
                        "200G.txt"), "r") as file:
    i = 0
    for line in file:
        energy, _= line.split("\n")
        coarse_energy_grid[i] = float(energy)
        i += 1


# Extracting xGPT sensitivities
xGPT_reader = serpentTools.read(os.path.join(os.getcwd(),
                                             "Desktop","Mémoire", 
                                             'MScThesis_2023_2024', "Code", "xGPT&GPT",  
                                             "XGPT", "main_sens0.m"))

sensitivities = xGPT_reader.energyIntegratedSens["keff"][0][0]
xGPT_vector = np.zeros((82,))
perts = xGPT_reader.perts

i = -1
for sensitivity, error in sensitivities:
    xGPT_vector[i] = sensitivity
    i += 1


# Extracting the eigen basis (Fission XS perturbation)
eigen_basis = np.zeros((82, 1500))
for label, index in perts.items():
    if index == 0:
        continue
    with open(os.path.join(os.getcwd(),
                           "Desktop","Mémoire", 'MScThesis_2023_2024',
                           "Code", "xGPT&GPT",
                           "XGPT", "{}.txt".format(label))) as file:
        lines = file.readlines()
        i = 0
        for line in lines[2:]:
            energy, basis_coeff = line.split(" ")
            basis_coeff, _ = basis_coeff.split("\n")
            eigen_basis[index - 1][i] = float(basis_coeff)
            i += 1
            
# Extracting GPT sensitivities
GPT_reader = serpentTools.read(os.path.join(os.getcwd(),
                                            "Desktop","Mémoire", 
                                            'MScThesis_2023_2024', "Code", "xGPT&GPT",
                                            "GPT", "FC_Tf_1073_Tc_1073_sens0.m"))

zai_index = GPT_reader.zais[942390] # zai of Pu239
fission_index = GPT_reader.perts["mt 18 xs"] # Fission xs perturbation
el_scatter_index = GPT_reader.perts["mt 2 xs"] # Elastic scattering xs perturbation
rad_capt_index = GPT_reader.perts["mt 102 xs"] # Radiative capture xs perturbation

fission_sensitivities = GPT_reader.sensitivities["keff"][0][zai_index][fission_index]
el_scatter_sensitivities = GPT_reader.sensitivities["keff"][0][zai_index][el_scatter_index]
rad_capt_sensitivities = GPT_reader.sensitivities["keff"][0][zai_index][rad_capt_index]


GPT_vector = {"MT2": np.zeros((200,)), "MT18": np.zeros((200,)), "MT102": np.zeros((200,))}

for sensitivities, label in [(fission_sensitivities, "MT18"), 
                             (el_scatter_sensitivities, "MT2"),
                             (rad_capt_sensitivities, "MT102")]:
    i = 0
    for sensitivity, error in sensitivities:
        GPT_vector[label][i] = sensitivity
        i += 1

# Filling the diagonal of the singular matrix
singular_value_matrix = {"MT2": [], "MT18":[], "MT102": []}
i = 0
for label, index in perts.items():
    if index > 0:
        index = index - 1
        zai, mt, number = label.split("_")
        _, MT_num = mt.split("MT")
        file_name = os.path.join(os.getcwd(),
                                            "Desktop","Mémoire", 
                                            'MScThesis_2023_2024', "Code", "xGPT&GPT",
                                            "XGPT", "SVs_Pu-239_{}.txt".format(MT_num))
        label = "MT{}".format(MT_num)
        with open(file_name, 'r') as file:
            lines = file.readlines()
            sv, _ = lines[ int(number) - 1 ].split("\n")
            singular_value_matrix[label].append(float(sv))
        i += 1

covariance_matrix = {"MT2": np.zeros((200, 200)), "MT18": np.zeros((200, 200)), "MT102": np.zeros((200, 200))}

# -------------------------------------- UTILS ---------------------------------------

def down_binning(GPT_sensitivities, GA_energy_grid, coarse_energy_grid):
    downbinned = {}
    for label, sensitivities in GPT_sensitivities.items():
        downbinned[label] = evaluate_on_GA(sensitivities, GA_energy_grid, coarse_energy_grid)
    return downbinned

def evaluate_on_GA(GPT_sensitivities, GA_energy_grid, coarse_energy_grid):
    sensitivities_evaluated = np.zeros((len(GA_energy_grid) + 1,))
    j = 0
    energies = []
    for i in range(len(GPT_sensitivities)):

        cut = len(GPT_sensitivities) if j >= len(GA_energy_grid) else GA_energy_grid[j]
        prev_cut = 0 if j == 0 else GA_energy_grid[j - 1]

        if i >= prev_cut and i < cut:
            sensitivities_evaluated[j] += GPT_sensitivities[i]*(coarse_energy_grid[i+1]-coarse_energy_grid[i])
            energies.append(coarse_energy_grid[i+1]-coarse_energy_grid[i])

        else :
            sensitivities_evaluated[j] /= np.sum(energies) 

            j += 1
            energies = []
            cut = len(GPT_sensitivities) if j >= len(GA_energy_grid) else GA_energy_grid[j]
            prev_cut = 0 if j == 0 else GA_energy_grid[j - 1]
            sensitivities_evaluated[j] += GPT_sensitivities[i]*(coarse_energy_grid[i+1]-coarse_energy_grid[i])

            energies.append(coarse_energy_grid[i+1]- coarse_energy_grid[i])

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
def up_binning(fine_energy_grid, coarse_energy_grid, down_binned_vector):
    upbinned = {}
    for label, sens in down_binned_vector.items():
        upbinned[label] = extend(fine_energy_grid, coarse_energy_grid, sens)
    return upbinned

def extend(fine_energy_grid, coarse_energy_grid, down_binned_vector):
    up_binned_vector = np.zeros((len(fine_energy_grid)-1,))
    j = 1
    for i in range(1, len(fine_energy_grid)):
        prev_energy = coarse_energy_grid[j - 1]
        energy = coarse_energy_grid[j]

        if fine_energy_grid[i] >= prev_energy and fine_energy_grid[i] <= energy:
            up_binned_vector[i - 1] = down_binned_vector[j - 1]
            if fine_energy_grid[i] == energy:
                j += 1
        else:
            SL = down_binned_vector[j - 1]
            SR = down_binned_vector[j]
            DEL = (energy - fine_energy_grid[i - 1])
            DER = (fine_energy_grid[i] - energy)
            up_binned_vector[i-1] = (SL * DEL + SR*DER)/(DEL + DER)
            j += 1

    return up_binned_vector

# Projection onto the eigen basis 
def projection(dict_vector, eigen_basis, labels=perts, energy_grid = fine_energy_grid):
    projected_vector = np.zeros((len(eigen_basis),))
    delta_E = np.diff(energy_grid)
    for label, index in labels.items():
        if index == 0:
            continue
        _, label, _ = label.split("_")
        vector = dict_vector[label]
        
        projected_vector[index-1] = trapz(np.array(vector)*np.array(eigen_basis[index-1]), energy_grid[:-1])

    return projected_vector


# Comparing xGPT et evaluated GPT
def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity

#  -------------------------------------- DEFINITION OF THE 3 FITNESS FUNCTIONS  -------------------------------------- 

def compare_vectors_GPT_xGPT(GA_grid):
    projected_GPT = get_GPT_on_eigenbasis(GA_grid)
    projected_GPT = (projected_GPT - np.mean(projected_GPT))/np.std(projected_GPT)

    XGPT_vector = (xGPT_vector - np.mean(xGPT_vector))/np.std(xGPT_vector)
    return 1 - cosine_similarity(projected_GPT , XGPT_vector)
 
def get_GPT_on_eigenbasis(GA_grid):
    GPT_GA = down_binning(GPT_vector, GA_grid, coarse_energy_grid)
    GA_energy_grid = energy_from_energy_grid(coarse_energy_grid, GA_grid)
    extended_GPT_GA = up_binning(fine_energy_grid, GA_energy_grid, GPT_GA)
    projected_GPT = projection(extended_GPT_GA, eigen_basis)
    return projected_GPT


def compare_uncertainty_GPT_xGPT(GA_grid):
    projected_GPT = get_GPT_on_eigenbasis(GA_grid)
    uncertainty_GPT = projected_GPT.T @ projected_GPT
    uncertainty_xGPT = xGPT_vector.T @ xGPT_vector
    return abs(uncertainty_GPT-uncertainty_xGPT)

def compute_uncertainty_GPT(GA_grid):
    GPT_GA = down_binning(GPT_vector, GA_grid, coarse_energy_grid)
    GA_energy_grid = energy_from_energy_grid(coarse_energy_grid, GA_grid)
    extended_GPT_GA = up_binning(coarse_energy_grid, GA_energy_grid, GPT_GA)
    uncertainty_evaluated_GPT , uncertainty_GPT = np.zeros((len(extended_GPT_GA), )), np.zeros((len( extended_GPT_GA ),))

    for label in extended_GPT_GA.keys():

        uncertainty_evaluated_GPT[i] = extended_GPT_GA[label].T @ covariance_matrix[label] @ extended_GPT_GA[label]
        uncertainty_GPT[i] = GPT_vector[label].T @ covariance_matrix[label] @ GPT_vector[label]

    return np.linalg.norm(uncertainty_evaluated_GPT - uncertainty_GPT)

def compare_vectors_GPT(GA_grid):
    GPT_GA = down_binning(GPT_vector, GA_grid, coarse_energy_grid)
    GA_energy_grid = energy_from_energy_grid(coarse_energy_grid, GA_grid)
    extended_GPT_GA = up_binning(coarse_energy_grid, GA_energy_grid, GPT_GA)
    similarities = 0
    for label, evaluated_values in extended_GPT_GA.items():
        values = GPT_vector[label]
        similarities += cosine_similarity(evaluated_values, values)

    return 1 - similarities/len(extended_GPT_GA)


#  -------------------------------------- ENVISIONNING THE RESULTS  -------------------------------------- 

import matplotlib.pyplot as plt

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

    plt.step(np.arange(82), (projected_GPT - np.mean(projected_GPT))/np.std(projected_GPT), where="pre", label="evaluated")
    plt.step(np.arange(82),( xGPT_vector - np.mean(xGPT_vector))/np.std(xGPT_vector), where="pre", label="fine")

    plt.legend()
    plt.show()

def energy_cut_from_energy_grid(coarse_energy, energy_grid):
    return [np.abs(coarse_energy - energy).argmin() for energy in energy_grid][1:-1]


def compare_integrals(GA_grid):
    nGroups = len(GA_grid) + 1
    for label, item in GPT_vector.items():
        evaluated_GPT = evaluate_on_GA(GPT_vector[label], GA_grid, coarse_energy_grid)
        fine_integral = trapz(item, coarse_energy_grid[:-1])
        GA_energy_grid =  energy_from_energy_grid(coarse_energy_grid, GA_grid)
        downbinned_integral = trapz(evaluated_GPT, GA_energy_grid[:-1])

        fine_upbinned = extend(fine_energy_grid, GA_energy_grid, evaluated_GPT)
        coarse_upbinned = extend(coarse_energy_grid, GA_energy_grid, evaluated_GPT)
        upbinned_coarse_integral = trapz(coarse_upbinned, coarse_energy_grid[:-1])
        upbinned_fine_integral = trapz(fine_upbinned , fine_energy_grid[:-1])
        print("{} :\n fine integral value = (200G) {} \n down-binned integral value = ({}G) {} \n up-binned integral value = (200G) {}\n up-binned integral value = (1500G) {} \n".format(label, fine_integral, nGroups, downbinned_integral, upbinned_coarse_integral, upbinned_fine_integral))



ECCO33 = [1.9640330000E+01,1.0000000000E+01, 6.0653070000E+00, 3.6787940000E+00,
           2.2313020000E+00, 1.3533530000E+00, 8.2085000000E-01, 4.9787070000E-01, 3.0197380000E-01,
           1.8315640000E-01, 1.1109000000E-01, 6.7379470000E-02, 4.0867710000E-02, 2.4787520000E-02,
           1.5034390000E-02, 9.1188200000E-03, 5.5308440000E-03, 3.3546260000E-03, 2.0346840000E-03, 
           1.2340980000E-03, 7.4851830000E-04, 4.5399930000E-04, 3.0432480000E-04, 1.4862540000E-04,
           9.1660880000E-05, 6.7904050000E-05, 4.0169000000E-05, 2.2603290000E-05, 1.3709590000E-05,
           8.3152870000E-06, 4.0000000000E-06, 5.4000000000E-07, 1.0000000000E-07, 1.0000100000E-11]
