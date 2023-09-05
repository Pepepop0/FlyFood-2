import numpy as np
import matplotlib.pyplot as plt
import random

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

def ant_colony_optimization(points, n_ants, n_iterations, alpha, beta, evap_rate):
    best_path = None
    best_len = np.inf
    num_points = len(points)
    pheromones = np.ones( (num_points , num_points) )
    best_lengths = []

    for _ in range(n_iterations):
        paths = []
        paths_len = []

        for ant in range(n_ants):
            #Pode ser otimizado no futuro
            new_path , new_len = ant_move( pheromones , points, alpha, beta, num_points)
            paths.append(new_path)
            paths_len.append(new_len)

            if new_len < best_len:
                best_len = new_len
                best_path = new_path
            pheromones *= evap_rate
        best_lengths.append(best_len)

    return best_len , best_path, best_lengths

def get_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def ant_move(pheromones, points, alpha, beta, num_points):
    visited = [False] * num_points
    current_point = random.randint(0, num_points - 1)
    visited[current_point] = True
    path = [current_point]
    path_len = 0

    while False in visited:
        unvisited = np.where(np.logical_not(visited))[0]
        weights = np.zeros(len(unvisited))
        for i, unvisited_point in enumerate(unvisited):
            weights[i] = ((pheromones[current_point, unvisited_point] ** alpha) / (get_distance(points[current_point], points[unvisited_point]) ** beta))

        #Pode dar estouro
        total_weight = np.sum(weights)
        if total_weight == 0:
            # Evitar a divisão por zero
            weights = np.ones(len(unvisited)) / len(unvisited)
        else:
            # Adicionar valor mínimo para evitar divisões por zero
            epsilon = 1e-10  # Valor mínimo
            weights /= (total_weight + epsilon)


        next_point = random.choices(unvisited, weights=weights)[0]
        path.append(next_point)
        path_len += get_distance(points[current_point], points[next_point])
        visited[next_point] = True
        current_point = next_point

    path.append(path[0])
    path_len += get_distance(points[current_point], points[path[0]])
    return path, path_len


def plot_evolution(best_lengths, optimal_length):
    generations = range(len(best_lengths))
    
    plt.figure(figsize=(12, 6))
    plt.plot(generations, best_lengths, label='Melhor Comprimento')
    plt.axhline(y=optimal_length, color='r', linestyle='--', label='Comprimento Ótimo')
    
    plt.xlabel('Iterações')
    plt.ylabel('Comprimento')
    plt.title(f'Evolução do Comprimento da Rota \nMelhor rota: {best_lengths[len(best_lengths) - 1]:.2f}')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_route(points, route):
    x = [points[i][0] for i in route]
    y = [points[i][1] for i in route]
    
    plt.figure(figsize=(8, 8))
    plt.plot(x, y, marker='o', linestyle='-', markersize=5, color='b')
    plt.plot([x[0], x[-1]], [y[0], y[-1]], linestyle='-', color='b')
    
    for i, txt in enumerate(route):
        plt.annotate(txt, (x[i], y[i]), fontsize=8, ha='right')
    
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.title('Rota da Formiga')
    plt.grid(True)
    plt.show()

 
best_path_length, best_path, best_lengths = ant_colony_optimization(points= berlin52, n_ants= 10, n_iterations= 100, alpha=0.5, beta=2.5, evap_rate=0.8)
print(f"Best Path Length: {best_path_length:.2f}")
# Plotar a evolução do comprimento da rota
optimal_length = 7540
plot_evolution(best_lengths, optimal_length)

# Plotar a rota da formiga
plot_route(berlin52, best_path)
