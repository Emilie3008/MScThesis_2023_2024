import numpy as np
import os
import serpentTools

main_path = os.path.join(os.getcwd(), 
                       "Desktop","Mémoire", 'MScThesis_2023_2024',
                       "Code", "xGPT&GPT"
                       )

with open(os.path.join(main_path, "isotope.txt"), "r") as file:
    ISOTOPE = file.readlines()[0]

XGPT_energy_grid = np.zeros((1501,))
GPT_energy_grid = np.zeros((201,))

with open(os.path.join(main_path, "1500G.txt"), "r") as file:
    i = 0
    for line in file:
        energy, _= line.split("\n")
        XGPT_energy_grid[i] = float(energy)
        i += 1

with open(os.path.join(main_path, "200G.txt"), "r") as file:
    i = 0
    for line in file:
        energy, _= line.split("\n")
        GPT_energy_grid[i] = float(energy)
        i += 1


# Extracting xGPT sensitivities
def extract_xGPT_sensitivity_coefficients(filepath):
    xGPT_reader = serpentTools.read(filepath)

    sensitivities = xGPT_reader.energyIntegratedSens["keff"][0][0]
    perts = xGPT_reader.perts
    xGPT_vector = np.zeros((len(perts)-1,))
    

    i = -1
    for sensitivity, error in sensitivities:
        xGPT_vector[i] = sensitivity
        i += 1
    return xGPT_vector, perts

def extract_eigenbasis(filepath, perts):
    eigenbasis = np.zeros((len(perts)-1, len(XGPT_energy_grid)-1))
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
    m = len(perts) - 1
    singular_matrix = np.zeros((m, m))
    
    for label, index in perts.items():
        if index > 0:
            index = index - 1
            zai, mt, number = label.split("_")
            _, MT_num = mt.split("MT")
            file_name = os.path.join(main_path, "XGPT", ISOTOPE, filename + MT_num + ".txt")
            label = "MT{}".format(MT_num)

            with open(file_name, 'r') as file:
                lines = file.readlines()
                sv, _ = lines[ int(number) - 1 ].split("\n")
                singular_matrix[index][index] = float(sv)
    return singular_matrix


def extract_data():
    filepath_GPT = os.path.join(main_path,"GPT", "FC_Tf_1073_Tc_1073_sens0.m")
    
    zai =  942390 if ISOTOPE == 'Pu239' else 922380
    GPT_vector = extract_GPT_sensitivity_coefficients(filepath_GPT, zai)

    filepath_xGPT =  os.path.join(main_path, "XGPT", ISOTOPE, "FC_Tf_1073_Tc_1073_sens0.m")
    xGPT_vector, perts = extract_xGPT_sensitivity_coefficients(filepath_xGPT)

    filepath_eigen_basis =os.path.join(main_path, "XGPT", ISOTOPE)
    eigenbasis = extract_eigenbasis(filepath_eigen_basis, perts)

    singular_value_file_name = 'SVs_Pu-239_' if ISOTOPE == "Pu239" else 'SVs_U-238_'
    singular_matrix = extract_singular_matrix(singular_value_file_name, perts)

    return (GPT_vector, xGPT_vector, perts, eigenbasis, singular_matrix)

GPT_vector, xGPT_vector, perts, eigenbasis, singular_matrix  = extract_data()


ECCO33 = sorted([1.9640330000E+01,1.0000000000E+01, 6.0653070000E+00, 3.6787940000E+00,
           2.2313020000E+00, 1.3533530000E+00, 8.2085000000E-01, 4.9787070000E-01, 3.0197380000E-01,
           1.8315640000E-01, 1.1109000000E-01, 6.7379470000E-02, 4.0867710000E-02, 2.4787520000E-02,
           1.5034390000E-02, 9.1188200000E-03, 5.5308440000E-03, 3.3546260000E-03, 2.0346840000E-03, 
           1.2340980000E-03, 7.4851830000E-04, 4.5399930000E-04, 3.0432480000E-04, 1.4862540000E-04,
           9.1660880000E-05, 6.7904050000E-05, 4.0169000000E-05, 2.2603290000E-05, 1.3709590000E-05,
           8.3152870000E-06, 4.0000000000E-06, 5.4000000000E-07, 1.0000000000E-07, 1.0000100000E-11])

SCALE56 = sorted(np.array([2.0000000000E+07, 6.4340000000E+06, 4.3040000000E+06, 3.0000000000E+06, 
                  1.8500000000E+06, 1.5000000000E+06, 1.2000000000E+06, 8.6110000000E+06,
                  7.5000000000E+05, 6.0000000000E+05, 4.7000000000E+05, 3.3000000000E+05,
                  2.7000000000E+05, 2.0000000000E+05, 5.0000000000E+04, 2.0000000000E+04, 
                  1.7000000000E+04, 3.7400000000E+03, 2.2500000000E+03, 1.9150000000E+02, 
                  1.8770000000E+02, 1.1750000000E+02, 1.1600000000E+02, 1.0500000000E+02, 
                  1.0120000000E+02, 6.7500000000E+01, 6.5000000000E+01, 3.7130000000E+01, 
                  3.6000000000E+01, 2.1750000000E+01, 2.1200000000E+01, 2.0500000000E+01, 
                  7.0000000000E+00, 6.8750000000E+00, 6.5000000000E+00, 6.2500000000E+00,
                  5.0000000000E+00, 1.1300000000E+00, 1.0800000000E+00, 1.0100000000E+00,
                  6.2500000000E-01, 4.5000000000E-01, 3.7500000000E-01, 3.5000000000E-01, 
                  3.2500000000E-01, 2.5000000000E-01, 2.0000000000E-01, 1.5000000000E-01,
                  1.0000000000E-01, 8.0000000000E-02, 6.0000000000E-02, 5.0000000000E-02,
                  4.0000000000E-02, 2.5300000000E-02, 1.0000000000E-02, 4.0000000000E-03])*1e-6)
