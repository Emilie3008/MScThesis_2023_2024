# MScThesis_2023_2024

1. DESCRIPTION DU PROJET

2. DESCRIPTION DU SAMPLE CASE

3. DESCRIPTION DU ENERGY GRID DESIGN MODULE
   
\begin{itemize}
    \item \textit{isotope.txt}: This text file should be filled with a single entry for the nuclide on which we wish to optimise the grids. If we want to optimise with respect to $Pu^{239}$, enter Pu239.  If we want to optimise for $U^{238}$, enter U238. 
    \item \textit{1500G.txt} and \textit{226G.txt} : These text files give the exact energies of the XGPT and GPT fine discretisations. Each line corresponds to a new energy. For an N-group discretisation, N + 1 energies must be defined. Thus, \textit{1500G.txt} contains 1501 lines and \textit{226G.txt} contains 227 lines.
    \item \textit{GPT} folder: This folder contains the results for GPT-scored sensitivities of $U^{238}$ and $Pu^{239}$ from the simulation with Serpent 2. 
    \item \textit{XGPT} folder:  This folder contains two sub-folders, one with Pu239 data and the other with U238 data. Each sub-folder contains the results of the sensitivities scored by XGPT from the simulation with Serpent 2, as well as text files containing the eigenfunctions evaluated on the 1500G and the singular values.
    \item \textit{Tests} folder: This file contains the results of the tests carried out in chapter \ref{verification}, as well as a python file \textit{test\_fitness\_functions.py} which tests the functions used to calculate the fitness functions.
    \item \textit{data\_extraction.py}: All data extraction and pre-processing operations are carried out by this script.
    \item \textit{fitness\_utils.py}: This Python file contains all the functions performing the various sub-steps (evaluation on a coarse grid, extension, projection, etc.) of the fitness functions.
    \item \textit{fitness\_functions.py}: This Python file contains the 4 fitness functions that have been developed.
    \item \textit{plot\_results.py}: This Python file contains all the functions used to plot the results of the optimisation (compare the integral sensitivity, compare the fine vs. evaluated sensitivity profile, etc.).
    \item \textit{genetic\_algorithm.py}: This Python file contains the code for the genetic algorithm. It is essentially the same code as the one used with the sample case, with minor details changing (e.g. different default parameters, fitness storage, etc.).
    \item \textit{main.py}: This Python file is used to run the simulation.
\end{itemize}
