import numpy as np
import matplotlib.pyplot as plt
import random

# Definindo as coordenadas do Berlin52
berlin52 = [
    (565, 575), (25, 185), (345, 750), (945, 685), (845, 655),
    (880, 660), (25, 230), (525, 1000), (580, 1175), (650, 1130), (1605, 620),
    (1220, 580), (1465, 200), (1530, 5), (845, 680), (725, 370), (145, 665),
    (415, 635), (510, 875), (560, 365), (300, 465), (520, 585), (480, 415),
    (835, 625), (975, 580), (1215, 245), (1320, 315), (1250, 400), (660, 180),
    (410, 250), (420, 555), (575, 665), (1150, 1160), (700, 580), (685, 595),
    (685, 610), (770, 610), (795, 645), (720, 635), (760, 650), (475, 960),
    (95, 260), (875, 920), (700, 500), (555, 815), (830, 485), (1170, 65),
    (830, 610), (605, 625), (595, 360), (1340, 725), (1740, 245)
]

class AIS_TSP_Solver:
    def __init__(self, coordinates, num_iterations=1000, num_antibodies=100, num_clones=5, num_mutation_rate = 0.2, num_selection_ratio = 0.5, num_clone_rate = 1.5):
        self.coordinates = coordinates
        self.num_iterations = num_iterations
        self.num_antibodies = num_antibodies
        self.num_clones = num_clones
        self.mutation_rate = num_mutation_rate
        self.selection_ratio = num_selection_ratio
        self.clone_rate = num_clone_rate
        
    @staticmethod
    def _distance(point1, point2):
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def _total_distance(self, route):
        return sum([self._distance(route[i], route[i+1]) for i in range(len(route)-1)]) + self._distance(route[-1], route[0])

    
    def _mutate(self, route):
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(len(route)), 2)
            route[idx1], route[idx2] = route[idx2], route[idx1]
        return route


    def solve(self):
        antibodies = [random.sample(self.coordinates, len(self.coordinates)) for _ in range(self.num_antibodies)]
        best_distance = float('inf')
        best_solution = None
        
        for iteration in range(self.num_iterations):
            antibodies.sort(key=self._total_distance)
            selected_antibodies = antibodies[:int(self.num_antibodies * self.selection_ratio)]  
            
            best_current = antibodies[0]
            
            if self._total_distance(best_current) < best_distance:
                best_distance = self._total_distance(best_current)
                best_solution = best_current[:]
            
            new_population = antibodies[:self.num_antibodies // 2]
            for ab in antibodies[:self.num_clones]:
                for _ in range(int(self.clone_rate * self.num_clones)):
                    clone = ab[:]
                    new_population.append(self._mutate(clone))
            
            antibodies = new_population
        
        return best_solution, best_distance

    def solve_with_evolution(self):
        antibodies = [random.sample(self.coordinates, len(self.coordinates)) for _ in range(self.num_antibodies)]
        best_distances_evolution = []
        best_distance = float('inf')
        best_solution = None
        
        for iteration in range(self.num_iterations):
            antibodies.sort(key=self._total_distance)
            best_current = antibodies[0]
            
            current_best_distance = self._total_distance(best_current)
            best_distances_evolution.append(current_best_distance)
            
            if current_best_distance < best_distance:
                best_distance = current_best_distance
                best_solution = best_current[:]
            
            new_population = antibodies[:self.num_antibodies // 2]
            for ab in antibodies[:self.num_clones]:
                for _ in range(self.num_clones):
                    clone = ab[:]
                    new_population.append(self._mutate(clone))
            
            antibodies = new_population
        
        return best_solution, best_distance, best_distances_evolution




# Define the optimal route value for Berlin52
optimal_route_value = 7540

def compute_total_distance(route):
    """Compute the total distance of a route using the provided _distance function."""
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += AIS_TSP_Solver._distance(route[i], route[i + 1])
    total_distance += AIS_TSP_Solver._distance(route[-1], route[0])  # Return to the starting point
    return total_distance

def plot_route(route, title="Melhor Rota AIS para Berlin52"):
    """Plot the route and display the total distance in the title."""
    x, y = zip(*route)
    x = x + (x[0],)
    y = y + (y[0],)
    
    total_distance = compute_total_distance(route)
    title += f" - Distância: {total_distance:.2f}"
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, '-o')
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

def plot_evolution(evolution):
    """Plot the evolution of the route distance and display a line for the optimal value."""
    
    plt.figure(figsize=(10, 6))
    plt.plot(evolution, label="Rota AIS")
    plt.axhline(optimal_route_value, color='r', linestyle='--', label="Rota Ótima")
    plt.title("Evolução da Melhor Rota com as Gerações")
    plt.xlabel("Geração")
    plt.ylabel("Distância da Rota")
    plt.legend()
    plt.grid(True)
    plt.show()


print("Iniciando a otimização da rota...")
solver = AIS_TSP_Solver(berlin52, num_iterations=1000, num_antibodies=1000, num_clones=8, num_mutation_rate = 0.78, num_selection_ratio = 0.33, num_clone_rate = 1.39)
best_route, best_dist, evolution = solver.solve_with_evolution()
print(best_dist)
print("Otimização da rota concluída.")
plot_route(best_route)
plot_evolution(evolution)