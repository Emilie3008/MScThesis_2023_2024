import numpy as np
import os
import serpentTools


main_path = os.path.join(os.getcwd(), 
                       "Desktop","Mémoire", 'MScThesis_2023_2024',
                       "Code", "xGPT&GPT"
                       )

with open(os.path.join(main_path, "isotope.txt"), "r") as file:
    ISOTOPE = file.readlines()[0]

fine_energy_grid = np.zeros((1501,))
coarse_energy_grid = np.zeros((201,))

with open(os.path.join(main_path, "1500G.txt"), "r") as file:
    i = 0
    for line in file:
        energy, _= line.split("\n")
        fine_energy_grid[i] = float(energy)
        i += 1


with open(os.path.join(main_path, "200G.txt"), "r") as file:
    i = 0
    for line in file:
        energy, _= line.split("\n")
        coarse_energy_grid[i] = float(energy)
        i += 1


# Extracting xGPT sensitivities
def extract_xGPT_sensitivity_coefficients(filepath):
    xGPT_reader = serpentTools.read(filepath)

    sensitivities = xGPT_reader.energyIntegratedSens["keff"][0][0]
    xGPT_vector = np.zeros((82,))
    perts = xGPT_reader.perts

    i = -1
    for sensitivity, error in sensitivities:
        xGPT_vector[i] = sensitivity
        i += 1
    return xGPT_vector, perts

def extract_eigenbasis(filepath, perts):
    eigenbasis = np.zeros((82, 1500))
    for label, index in perts.items():
        if index == 0:
            continue
        filename = os.path.join(filepath, label+'.txt')
        with open(filename) as file:
            lines = file.readlines()
            i = 0
            for line in lines[2:]:
                energy, basis_coeff = line.split(" ")
                basis_coeff, _ = basis_coeff.split("\n")
                eigenbasis[index - 1][i] = float(basis_coeff)
                i += 1
    return eigenbasis


def extract_GPT_sensitivity_coefficients(filepath, zai):

    # Extracting GPT sensitivities
    GPT_reader = serpentTools.read(filepath)

    zai_index = GPT_reader.zais[zai] 

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

    return GPT_vector


def extract_singular_matrix(filename, perts):
    singular_matrix_Pu = {"MT2": [], "MT18":[], "MT102": []}
    i = 0
    for label, index in perts.items():
        if index > 0:
            index = index - 1
            zai, mt, number = label.split("_")
            _, MT_num = mt.split("MT")
            file_name = os.path.join(main_path, "XGPT", filename + MT_num)
            label = "MT{}".format(MT_num)
            with open(file_name, 'r') as file:
                lines = file.readlines()
                sv, _ = lines[ int(number) - 1 ].split("\n")
                singular_matrix_Pu[label].append(float(sv))
            i += 1

def extract_covariance_matrix(isotope, energy_grid):
    return {"MT2": np.zeros((200, 200)), "MT18": np.zeros((200, 200)), "MT102": np.zeros((200, 200))}

def extract_data():
    zai = 942390 if ISOTOPE == "Pu239" else 922380

    filepath_GPT = os.path.join(main_path,"GPT", "FC_Tf_1073_Tc_1073_sens0.m")
    GPT_vector = extract_GPT_sensitivity_coefficients(filepath_GPT, zai)

    filepath_xGPT =  os.path.join(main_path, "XGPT", ISOTOPE, "FC_Tf_1073_Tc_1073_sens0.m")
    xGPT_vector, perts = extract_xGPT_sensitivity_coefficients(filepath_xGPT)

    filepath_eigen_basis =os.path.join(main_path,"XGPT", ISOTOPE)
    eigenbasis = extract_eigenbasis(filepath_eigen_basis, perts)
    return GPT_vector, xGPT_vector, perts, eigenbasis

ECCO33 = [1.9640330000E+01,1.0000000000E+01, 6.0653070000E+00, 3.6787940000E+00,
           2.2313020000E+00, 1.3533530000E+00, 8.2085000000E-01, 4.9787070000E-01, 3.0197380000E-01,
           1.8315640000E-01, 1.1109000000E-01, 6.7379470000E-02, 4.0867710000E-02, 2.4787520000E-02,
           1.5034390000E-02, 9.1188200000E-03, 5.5308440000E-03, 3.3546260000E-03, 2.0346840000E-03, 
           1.2340980000E-03, 7.4851830000E-04, 4.5399930000E-04, 3.0432480000E-04, 1.4862540000E-04,
           9.1660880000E-05, 6.7904050000E-05, 4.0169000000E-05, 2.2603290000E-05, 1.3709590000E-05,
           8.3152870000E-06, 4.0000000000E-06, 5.4000000000E-07, 1.0000000000E-07, 1.0000100000E-11]