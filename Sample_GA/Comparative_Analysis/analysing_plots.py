import numpy as np
import os 
import matplotlib.pyplot as plt

def analyse_file(file_path):
    runtime = np.zeros((100,))
    iterations = np.zeros((100,))
    with open(file_path, "r") as file:
        for i in range(100):
            line = file.readline()
            generation, time, _ = line.split(", ")
            iterations[i] = int(generation)
            runtime[i] = float(time)
    return runtime, iterations

def convergence_curves(file_path, nb_iterations):
    generations = np.zeros((int(nb_iterations),))
    fitness = np.zeros((int(nb_iterations),))
    with open(file_path, 'r') as file :
        for i in range(int(nb_iterations)):
            line = file.readline()
            gen, best_fitness = line.split(", ")
            best_fitness = best_fitness.split("\n")[0]
            generations[i] = int(gen)
            fitness[i] = float(best_fitness)
    return generations, fitness


# ===== ANALYSING THE SIMPLE CASE RESULTS =====   
simple_filepath = os.path.join(os.getcwd(), "Simple", "Simple_GA.txt")
simple_time, simple_iterations = analyse_file(simple_filepath)
s_index_min, s_index_max = np.argmin(simple_iterations), np.argmax(simple_iterations)
filepath_min = os.path.join(os.getcwd(), "Simple", "simple_{}.txt".format(s_index_min))
filepath_max = os.path.join(os.getcwd(), "Simple", "simple_{}.txt".format(s_index_max))
generation_simple_min, fitness_simple_min = convergence_curves(filepath_min, simple_iterations[s_index_min])
generation_simple_max, fitness_simple_max = convergence_curves(filepath_max, simple_iterations[s_index_max])

print("----- Results for the simple GA -----")
print("Mean iterations : {} +- {}".format(np.mean(simple_iterations), np.std(simple_iterations)))
print("Mean runtime (s) : {} +- {}".format(np.mean(simple_time), np.std(simple_time)))
print("Best case : convergence in {} iterations for the {}-th case ".format(simple_iterations[s_index_min], s_index_min))
print("Worst case : convergence in {} iterations for the {}-th case ".format(simple_iterations[s_index_max], s_index_max))
print("\n")

# ===== ANALYSING THE ADAPTIVE CASE RESULTS ===== 
adaptive_filepath = os.path.join(os.getcwd(), "Adaptive", "Adaptive_GA.txt")
adaptive_time, adaptive_iterations = analyse_file(adaptive_filepath)
a_index_min, a_index_max = np.argmin(adaptive_iterations), np.argmax(adaptive_iterations)
filepath_min = os.path.join(os.getcwd(), "Adaptive", "adaptive_{}.txt".format(a_index_min))
filepath_max = os.path.join(os.getcwd(), "Adaptive", "adaptive_{}.txt".format(a_index_max))
generation_adaptive_min, fitness_adaptive_min = convergence_curves(filepath_min, adaptive_iterations[a_index_min])
generation_adaptive_max, fitness_adaptive_max = convergence_curves(filepath_max, adaptive_iterations[a_index_max])

print("----- Results for the adaptive GA -----")
print("Mean iterations : {} +- {}".format(np.mean(adaptive_iterations), np.std(adaptive_iterations)))
print("Mean runtime (s) : {} +- {}".format(np.mean(adaptive_time), np.std(adaptive_time)))
print("Best case : convergence in {} iterations for the {}-th case ".format(adaptive_iterations[a_index_min], a_index_min))
print("Worst case : convergence in {} iterations for the {}-th case ".format(adaptive_iterations[a_index_max], a_index_max))
print("\n")

# ===== ANALYSING THE TOURNAMENT CASE RESULTS ===== 
tournament_filepath = os.path.join(os.getcwd(), "Tournaments", "Tournament_GA.txt")
tournament_time, tournament_iterations = analyse_file(tournament_filepath)
t_index_min, t_index_max = np.argmin(tournament_iterations), np.argmax(tournament_iterations)
filepath_min = os.path.join(os.getcwd(), "GA_RESULTS", "tournaments_{}.txt".format(t_index_min))
filepath_max = os.path.join(os.getcwd(), "GA_RESULTS", "tournaments_{}.txt".format(t_index_max))
generation_tournament_min, fitness_tournament_min = convergence_curves(filepath_min, tournament_iterations[t_index_min])
generation_tournament_max, fitness_tournament_max = convergence_curves(filepath_max, tournament_iterations[t_index_max])

print("----- Results for the tournament selection GA -----")
print("Mean iterations : {} +- {}".format(np.mean(tournament_iterations), np.std(tournament_iterations)))
print("Mean runtime (s) : {} +- {}".format(np.mean(tournament_time), np.std(tournament_time)))
print("Best case : convergence in {} iterations for the {}-th case ".format(tournament_iterations[t_index_min], t_index_min))
print("Worst case : convergence in {} iterations for the {}-th case ".format(tournament_iterations[t_index_max], t_index_max))
print("\n")

# ===== ANALYSING THE MULTI-PARENT CASE RESULTS ===== 
multi_filepath = os.path.join(os.getcwd(), "GA_RESULTS", "Multi", "Multi_GA.txt")
multi_time, multi_iterations = analyse_file(multi_filepath)
m_index_min, m_index_max = np.argmin(multi_iterations), np.argmax(multi_iterations)
filepath_min = os.path.join(os.getcwd(), "Multi", "multi_{}.txt".format(m_index_min))
filepath_max = os.path.join(os.getcwd(), "Multi", "multi_{}.txt".format(m_index_max))
generation_multi_min, fitness_multi_min = convergence_curves(filepath_min, multi_iterations[m_index_min])
generation_multi_max, fitness_multi_max = convergence_curves(filepath_max, multi_iterations[m_index_max])

print("----- Results for the multi-parent reproduction GA -----")
print("Mean iterations : {} +- {}".format(np.mean(multi_iterations), np.std(multi_iterations)))
print("Mean runtime (s) : {} +- {}".format(np.mean(multi_time), np.std(multi_time)))
print("Best case : convergence in {} iterations for the {}-th case ".format(multi_iterations[m_index_min], m_index_min))
print("Worst case : convergence in {} iterations for the {}-th case ".format(multi_iterations[m_index_max], m_index_max))
print("\n")

# ===== ANALYSING THE COMBINED CASE RESULTS ===== 
combined_filepath = os.path.join(os.getcwd(), "Combined", "Combined_GA.txt")
combined_time , combined_iterations = analyse_file(combined_filepath)
c_index_min, c_index_max = np.argmin(combined_iterations), np.argmax(combined_iterations)
filepath_min = os.path.join(os.getcwd(), "Combined", "combined_{}.txt".format(c_index_min))
filepath_max = os.path.join(os.getcwd(), "Combined", "combined_{}.txt".format(c_index_max))
generation_combined_min, fitness_combined_min = convergence_curves(filepath_min, combined_iterations[c_index_min])
generation_combined_max, fitness_combined_max = convergence_curves(filepath_max, combined_iterations[c_index_max])

print("----- Results for the combined GA -----")
print("Mean iterations : {} +- {}".format(np.mean(combined_iterations), np.std(combined_iterations)))
print("Mean runtime (s) : {} +- {}".format(np.mean(combined_time), np.std(combined_time)))
print("Best case : convergence in {} iterations for the {}-th case ".format(combined_iterations[c_index_min], c_index_min))
print("Worst case : convergence in {} iterations for the {}-th case ".format(combined_iterations[c_index_max], c_index_max))


# ===== PLOTS FOR THE WORST CASE SCENARIO =====

plt.figure(figsize=(10, 6))
for generation, fitness, label in [(generation_adaptive_max, fitness_adaptive_max, "Adaptive mutation rate"),
                            (generation_combined_max, fitness_combined_max, "Combined Case"),
                            (generation_multi_max, fitness_multi_max, "Multi-parent (10) reproduction"),
                            (generation_tournament_max, fitness_tournament_max, "Stochastic tournament selection"),
                            (generation_simple_max, fitness_simple_max, "Simple genetic algorithm")]:
    
    plt.plot(generation, fitness, label=label, linewidth=2.5)

plt.xlabel("Iterations")
plt.ylabel("Fitness of the best performing individual of the population")
plt.title("Worst case convergence")
plt.legend()
plt.show()

# ===== PLOTS FOR THE BEST CASE SCENARIO =====

plt.figure(figsize=(10, 6))
for generation, fitness, label in [(generation_adaptive_min, fitness_adaptive_min, "Adaptive mutation rate"),
                            (generation_combined_min, fitness_combined_min, "Combined Case"),
                            (generation_multi_min, fitness_multi_min, "Multi-parent (10) reproduction"),
                            (generation_tournament_min, fitness_tournament_min, "Stochastic tournament selection"),
                            (generation_simple_min, fitness_simple_min, "Simple genetic algorithm")]:
    

    plt.plot(generation, fitness, label=label, linewidth=2.5)

plt.xlabel("Iterations")
plt.ylabel("Fitness of the best performing individual of the population")
plt.title("Best case convergence")
plt.legend()
plt.show()
