# Energy Grid Design For Fast Reactors Sensitivity Calculations 

This code was developed as a part of my master science thesis in physics engineering at ULB, in a collaboration with the Belgian Nuclear Center SCK CEN. 

## Pre-requisites

This code was developped using Python version 12.3.0, as well as the following libraries and their versions:
- numpy version 1.26.4
- serpentTools version 0.10.1
- pandas version 1.3.2
- iracepy version 0.0.1, with R 4.4.0 installed

## Abstract

In this work, we explore the possibility of using genetic algorithms (GA) as a search method for the design of energy grids used in the framework of fast reactors sensitivity calculations.

The main idea behind the three proposed fitness functions is to measure how good a sensitivity vector evaluated on a few-group discretisations is at accurately representing the same sensitivity vector, but scored on a many-group energy discretisations.

On the one hand, the possibility of computing the many-groups sensitivities with Generalised Perturbation Theory (GPT), or with its recently updated version eXtended GPT (XGPT), is explored. On the other hand, we investigate the possibility of assessing the similarity between the many- and few-group scored sensitivity vectors through a cosine similarity or by comparing the variance on the response.

 The resulting energy grids are optimised for the nuclear system ALFRED, on the sensitivities of $k_{eff}$ for $Pu^{239}$ and $U^{238}$, the nuclides contributing the most to the overall uncertainties. The energy grids are analysed qualitatively and quantitatively. The proposed approach results in energy grids that maximise the similarity with the fine sensitivity vectors but do not significantly impact the measure of the representativity. 

## Sample Case
This sample case was used in the early stages of the development of the project in order to develop the genetic algorithm. Here is a brief description of the content of this folder :
- **Sample_Case_GA.py**: Contains the Genetic Algorithm used in the sample case. It features a multi-parent reproduction, an adaptive mutation rate and a stochastic tournament selection.
- **ComparativeAnalysis**: Contains the convergence curves of the comparative analysis of the different configurations
- **Irace Parametrisation**: Contains the scripts used to configure the parametrisation of the GA for the sample case.

## Energy grid design

- **ColorPalette** : I chose to associate to each eigenfunction a specific color in order to make the figures of Chapter 5 consistent. This folder contains textfiles with the RGB representation of the color of each eigenfunction.
- **CovarianceMatrices**: This folder contains the covariance matrices of $Pu^{239}$ and  $U^{238}$, processed over each energy grids of the subfolder **Grids**. Those covariance matrices were used for the comparisons of the representativities of Chapter 5.
- **GPT** folder: This folder contains the results for GPT-scored sensitivities of $U^{238}$ and $Pu^{239}$ from the simulation with Serpent 2.
- **XGPT** folder: This folder contains two sub-folders, one with Pu239 data and the other with U238 data. Each sub-folder contains the results of the sensitivities
- **Results** : This folder contains the results of the tests carried out in Chapter 4 and 5. It has 3 subfolders:
  - **ConvergenceCurves**: this subfolder contains the results of the tests of convergence for the different configuration of parameters.
  - **Pu239**: This folder contains the energy grids optimised on the sensitivity profiles of  $Pu^{239}$
  - **U238**: This folder contains the energy grids optimised on the sensitivity profiles of  $Pu^{239}$
    
- **extract_input_data.py**: All data extraction and pre-processing operations are carried out by this script.
- **extract_results.py**: Contains the function used to read and extract the best resulting energy grids.
- **fitness_functions.py**: This Python file contains the 4 fitness functions that have been developed.
- **fitness_utils.py**: This Python file contains all the functions performing the various sub-steps (evaluation on a coarse grid, extension, projection, etc.) of the fitness functions.
- **isotope.txt**: This text file should be filled with a single entry for the nuclide on which we wish to optimise the grids. If we want to optimise with respect to $Pu^{239}$, enter Pu239. If we want to optimise for $U^{238}$, enter U238.
- **1500G.txt** and **226G.txt**: These text files give the exact energies of the XGPT and GPT fine discretisations. Each line corresponds to a new energy. For an N-group discretisation, N + 1 energies must be defined. Thus, **1500G.txt** contains 1501 lines and **226G.txt** contains 227 lines.
scored by XGPT from the simulation with Serpent 2, as well as text files containing the eigenfunctions evaluated on the 1500G and the singular values
- **plot_results.py**: This Python file contains all the functions used to plot the results of the optimisation (plot the different eigenfunctions, compare the fine vs. evaluated sensitivity profile, etc.).
- **genetic_algorithm.py**: This Python file contains the code for the genetic algorithm. It is essentially the same code as the one used with the sample case, with minor details changing (e.g. different default parameters, fitness storage, etc.).
- **main.py**: This Python file is used to run the simulation.


