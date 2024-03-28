import numpy as np
import os 
import matplotlib.pyplot as plt

simple_time = np.zeros((100,))
simple_iterations = np.zeros(100,)
with open(os.path.join(os.getcwd(), "Simple", "Simple_GA.txt"), 'r') as file :
    for i in range(100):
        line = file.readline()
        generation, time, _ = line.split(", ")
        simple_iterations[i] = int(generation)
        simple_time[i] = float(time)
print("----- Results for the simple GA -----")
print(np.mean(simple_iterations), np.std(simple_iterations))
print(np.mean(simple_time), np.std(simple_time))
s_index_min = np.argmin(simple_iterations)
print("Minimum ", s_index_min, simple_iterations[s_index_min])
s_index_max = np.argmax(simple_iterations)
print("Maximum ", s_index_max, simple_iterations[s_index_max])
print("\n")

generations_simple_max = np.zeros((int(simple_iterations[s_index_max]),))
fitness_simple_max = np.zeros((int(simple_iterations[s_index_max]),))
with open(os.path.join(os.getcwd(), "Simple", "simple_{}.txt".format(s_index_max)), 'r') as file :
    for i in range(int(simple_iterations[s_index_max])):
        line = file.readline()
        gen, best_fitness = line.split(", ")
        best_fitness = best_fitness.split("\n")[0]
        generations_simple_max[i] = int(gen)
        fitness_simple_max[i] = float(best_fitness)

generations_simple_min = np.zeros((int(simple_iterations[s_index_min]),))
fitness_simple_min = np.zeros((int(simple_iterations[s_index_min]),))
with open(os.path.join(os.getcwd(), "Simple", "simple_{}.txt".format(s_index_min)), 'r') as file :
    for i in range(int(simple_iterations[s_index_min])):
        line = file.readline()
        gen, best_fitness = line.split(", ")
        best_fitness = best_fitness.split("\n")[0]
        generations_simple_min[i] = int(gen)
        fitness_simple_min[i] = float(best_fitness)

adaptive_time = np.zeros((100,))
adaptive_iterations = np.zeros(100,)
with open(os.path.join(os.getcwd(), "Adaptive", "Adaptive_GA.txt"), 'r') as file :
    for i in range(100):
        line = file.readline()
        generation, time, _ = line.split(", ")
        adaptive_iterations[i] = int(generation)
        adaptive_time[i] = float(time)

print("----- Results for the adaptive GA -----")
print(np.mean(adaptive_iterations), np.std(adaptive_iterations))
print(np.mean(adaptive_time), np.std(adaptive_time))
a_index_min = np.argmin(adaptive_iterations)
print("Minimum ", a_index_min, adaptive_iterations[a_index_min])
a_index_max = np.argmax(adaptive_iterations)
print("Maximum ", a_index_max, adaptive_iterations[a_index_max])
print("\n")

generations_adaptive_max = np.zeros((int(adaptive_iterations[a_index_max]),))
fitness_adaptive_max = np.zeros((int(adaptive_iterations[a_index_max]),))
with open(os.path.join(os.getcwd(), "Adaptive", "adaptive_{}.txt".format(a_index_max)), 'r') as file :
    for i in range(int(adaptive_iterations[a_index_max])):
        line = file.readline()
        gen, best_fitness = line.split(", ")
        best_fitness = best_fitness.split("\n")[0]
        generations_adaptive_max[i] = int(gen)
        fitness_adaptive_max[i] = float(best_fitness)

generations_adaptive_min = np.zeros((int(adaptive_iterations[a_index_min]),))
fitness_adaptive_min = np.zeros((int(adaptive_iterations[a_index_min]),))
with open(os.path.join(os.getcwd(), "Adaptive", "adaptive_{}.txt".format(a_index_min)), 'r') as file :
    for i in range(int(adaptive_iterations[a_index_min])):
        line = file.readline()
        gen, best_fitness = line.split(", ")
        best_fitness = best_fitness.split("\n")[0]
        generations_adaptive_min[i] = int(gen)
        fitness_adaptive_min[i] = float(best_fitness)


tournament_time = np.zeros((100,))
tournament_iterations = np.zeros(100,)
with open(os.path.join(os.getcwd(), "Tournaments", "Tournament_GA.txt"), 'r') as file :
    for i in range(100):
        line = file.readline()
        generation, time, _ = line.split(", ")
        tournament_iterations[i] = int(generation)
        tournament_time[i] = float(time)

print("----- Results for the tournament selection GA -----")
print(np.mean(tournament_iterations), np.std(tournament_iterations))
print(np.mean(tournament_time), np.std(tournament_time))
t_index_min = np.argmin(tournament_iterations)
print("Minimum ", t_index_min, tournament_iterations[t_index_min])
t_index_max = np.argmax(tournament_iterations)
print("Maximum ", t_index_max, tournament_iterations[t_index_max])
print("\n")

generations_tournament_max = np.zeros((int(tournament_iterations[t_index_max]),))
fitness_tournament_max = np.zeros((int(tournament_iterations[t_index_max]),))
with open(os.path.join(os.getcwd(), "Tournaments", "tournaments_{}.txt".format(t_index_max)), 'r') as file :
    for i in range(int(tournament_iterations[t_index_max])):
        line = file.readline()
        gen, best_fitness = line.split(", ")
        best_fitness = best_fitness.split("\n")[0]
        generations_tournament_max[i] = int(gen)
        fitness_tournament_max[i] = float(best_fitness)

generations_tournament_min = np.zeros((int(tournament_iterations[t_index_min]),))
fitness_tournament_min = np.zeros((int(tournament_iterations[t_index_min]),))
with open(os.path.join(os.getcwd(), "Tournaments", "tournaments_{}.txt".format(t_index_min)), 'r') as file :
    for i in range(int(tournament_iterations[t_index_min])):
        line = file.readline()
        gen, best_fitness = line.split(", ")
        best_fitness = best_fitness.split("\n")[0]
        generations_tournament_min[i] = int(gen)
        fitness_tournament_min[i] = float(best_fitness)

multi_time = np.zeros((100,))
multi_iterations = np.zeros(100,)
with open(os.path.join(os.getcwd(), "Multi", "Multi_GA.txt"), 'r') as file :
    for i in range(100):
        line = file.readline()
        generation, time, _ = line.split(", ")
        multi_iterations[i] = int(generation)
        multi_time[i] = float(time)

print("----- Results for the multi-parent reproduction GA -----")
print(np.mean(multi_iterations), np.std(multi_iterations))
print(np.mean(multi_time), np.std(multi_time))
m_index_min = np.argmin(multi_iterations)
print("Minimum ", m_index_min, multi_iterations[m_index_min])
m_index_max = np.argmax(multi_iterations)
print("Maximum ", m_index_max, multi_iterations[m_index_max])
print("\n")

generations_multi_max = np.zeros((int(multi_iterations[m_index_max]),))
fitness_multi_max = np.zeros((int(multi_iterations[m_index_max]),))
with open(os.path.join(os.getcwd(), "Multi", "multi_{}.txt".format(m_index_max)), 'r') as file :
    for i in range(int(multi_iterations[m_index_max])):
        line = file.readline()
        gen, best_fitness = line.split(", ")
        best_fitness = best_fitness.split("\n")[0]
        generations_multi_max[i] = int(gen)
        fitness_multi_max[i] = float(best_fitness)

generations_multi_min = np.zeros((int(multi_iterations[m_index_min]),))
fitness_multi_min = np.zeros((int(multi_iterations[m_index_min]),))
with open(os.path.join(os.getcwd(), "Multi", "multi_{}.txt".format(m_index_min)), 'r') as file :
    for i in range(int(multi_iterations[m_index_min])):
        line = file.readline()
        gen, best_fitness = line.split(", ")
        best_fitness = best_fitness.split("\n")[0]
        generations_multi_min[i] = int(gen)
        fitness_multi_min[i] = float(best_fitness)

combined_time = np.zeros((100,))
combined_iterations = np.zeros(100,)
with open(os.path.join(os.getcwd(), "Combined", "Combined_GA.txt"), 'r') as file :
    for i in range(100):
        line = file.readline()
        generation, time, _ = line.split(", ")
        combined_iterations[i] = int(generation)
        combined_time[i] = float(time)

print("----- Results for the combined GA -----")
print(np.mean(combined_iterations), np.std(combined_iterations))
print(np.mean(combined_time), np.std(combined_time))
c_index_min = np.argmin(combined_iterations)
print("Minimum ", c_index_min, combined_iterations[c_index_min])
c_index_max = np.argmax(combined_iterations)
print("Maximum ", c_index_max, combined_iterations[c_index_max])

generations_combined_max = np.zeros((int(combined_iterations[c_index_max]),))
fitness_combined_max = np.zeros((int(combined_iterations[c_index_max]),))
with open(os.path.join(os.getcwd(), "Combined", "combined_{}.txt".format(c_index_max)), 'r') as file :
    for i in range(int(combined_iterations[c_index_max])):
        line = file.readline()
        gen, best_fitness = line.split(", ")
        best_fitness = best_fitness.split("\n")[0]
        generations_combined_max[i] = int(gen)
        fitness_combined_max[i] = float(best_fitness)

generations_combined_min = np.zeros((int(combined_iterations[c_index_min]),))
fitness_combined_min = np.zeros((int(combined_iterations[c_index_min]),))
with open(os.path.join(os.getcwd(), "Combined", "combined_{}.txt".format(c_index_min)), 'r') as file :
    for i in range(int(combined_iterations[c_index_min])):
        line = file.readline()
        gen, best_fitness = line.split(", ")
        best_fitness = best_fitness.split("\n")[0]
        generations_combined_min[i] = int(gen)
        fitness_combined_min[i] = float(best_fitness)

largeur = 10  # en pouces
hauteur = 6  # en pouces

# Créer une nouvelle figure avec une taille personnalisée
plt.figure(figsize=(largeur, hauteur))

for generation, fitness, label in [(generations_adaptive_max, fitness_adaptive_max, "Adaptive mutation rate"),
                            (generations_combined_max, fitness_combined_max, "Combined Case"),
                            (generations_multi_max, fitness_multi_max, "Multi-parent (10) reproduction"),
                            (generations_tournament_max, fitness_tournament_max, "Stochastic tournament selection"),
                            (generations_simple_max, fitness_simple_max, "Simple genetic algorithm")]:
    
    plt.plot(generation, fitness, label=label, linewidth=2.5)

plt.xlabel("Iterations")
plt.ylabel("Fitness of the best performing individual of the population")
plt.title("Worst case convergence")
plt.legend()
plt.show()


largeur = 10  # en pouces
hauteur = 6  # en pouces

# Créer une nouvelle figure avec une taille personnalisée
plt.figure(figsize=(largeur, hauteur))

for generation, fitness, label in [(generations_adaptive_min, fitness_adaptive_min, "Adaptive mutation rate"),
                            (generations_combined_min, fitness_combined_min, "Combined Case"),
                            (generations_multi_min, fitness_multi_min, "Multi-parent (10) reproduction"),
                            (generations_tournament_min, fitness_tournament_min, "Stochastic tournament selection"),
                            (generations_simple_min, fitness_simple_min, "Simple genetic algorithm")]:
    

    plt.plot(generation, fitness, label=label, linewidth=2.5)

plt.xlabel("Iterations")
plt.ylabel("Fitness of the best performing individual of the population")
plt.title("Best case convergence")
plt.legend()
plt.show()
