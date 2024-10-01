import tsplib95
import random
import numpy as np


class GTSP:
    def __init__(self, clusters, cost_matrix):
        self.clusters = clusters
        self.cost_matrix = cost_matrix
        self.num_clusters = len(clusters)
        self.nodes = [node for cluster in clusters for node in cluster]

    def evaluate(self, individual):
        """Calcula o custo total do ciclo."""
        cluster_order, node_selection = individual
        tour_cost = 0
        node_selection = [
            node - 1 for node in node_selection
        ]  # Ajustar índices para matriz

        # Custo total entre clusters conforme a ordem de visita
        for i in range(self.num_clusters - 1):
            node_a = node_selection[cluster_order[i]]
            node_b = node_selection[cluster_order[i + 1]]
            tour_cost += self.cost_matrix[node_a][node_b]

        # Custo para retornar ao início
        first_node = node_selection[cluster_order[0]]
        last_node = node_selection[cluster_order[-1]]
        tour_cost += self.cost_matrix[last_node][first_node]

        return tour_cost


def initialize_population(gtsp, population_size):
    """Inicializa a população com base no artigo."""
    population = []
    for _ in range(population_size):
        cluster_order = random.sample(range(gtsp.num_clusters), gtsp.num_clusters)
        node_selection = []
        for cluster in gtsp.clusters:
            probabilities = calculate_selection_probabilities(cluster, gtsp.cost_matrix)
            selected_node = np.random.choice(cluster, p=probabilities)
            node_selection.append(selected_node)
        population.append((cluster_order, node_selection))
    return population


def calculate_selection_probabilities(cluster, cost_matrix):
    """Calcula probabilidades conforme o artigo."""
    distances = np.array([np.min(cost_matrix[node - 1]) for node in cluster])
    probabilities = 1 / (distances + 1e-9)
    probabilities /= probabilities.sum()
    return probabilities


def crossover(parent1, parent2):
    """Crossover descrito no artigo."""
    cut_point = random.randint(1, len(parent1[0]) - 2)
    child1_c = parent1[0][:cut_point] + [
        c for c in parent2[0] if c not in parent1[0][:cut_point]
    ]
    child2_c = parent2[0][:cut_point] + [
        c for c in parent1[0] if c not in parent2[0][:cut_point]
    ]
    child1_n, child2_n = parent1[1][:], parent2[1][:]
    return (child1_c, child1_n), (child2_c, child2_n)


def intra_cluster_mutation(individual, clusters):
    """Mutação intra-cluster."""
    cluster_order, node_selection = individual
    mutated_node_selection = node_selection[:]
    cluster_idx = random.randint(0, len(clusters) - 1)
    mutated_node_selection[cluster_idx] = random.choice(clusters[cluster_idx])
    return (cluster_order, mutated_node_selection)


def inter_cluster_mutation(individual):
    """Mutação inter-cluster."""
    cluster_order, node_selection = individual
    mutated_cluster_order = cluster_order[:]
    i, j = random.sample(range(len(cluster_order)), 2)
    mutated_cluster_order[i], mutated_cluster_order[j] = (
        mutated_cluster_order[j],
        mutated_cluster_order[i],
    )
    return (mutated_cluster_order, node_selection)


def selection(population, fitnesses, num_to_select):
    """Seleção por torneio."""
    selected_indices = []
    for _ in range(num_to_select):
        i, j = random.sample(range(len(population)), 2)
        selected_indices.append(i if fitnesses[i] < fitnesses[j] else j)
    return [population[i] for i in selected_indices]


def genetic_algorithm(gtsp, population_size=50, generations=400, mutation_rate=0.05):
    """Implementação do algoritmo descrito no artigo."""
    population = initialize_population(gtsp, population_size)
    for generation in range(generations):
        fitnesses = [gtsp.evaluate(ind) for ind in population]
        num_parents = population_size // 2
        parents = selection(population, fitnesses, num_parents)

        offspring = []
        for i in range(0, len(parents) - 1, 2):
            child1, child2 = crossover(parents[i], parents[i + 1])
            offspring.append(child1)
            offspring.append(child2)

        for i in range(len(offspring)):
            if random.random() < mutation_rate:
                offspring[i] = intra_cluster_mutation(offspring[i], gtsp.clusters)
            if random.random() < mutation_rate:
                offspring[i] = inter_cluster_mutation(offspring[i])

        population += offspring
        fitnesses += [gtsp.evaluate(ind) for ind in offspring]
        best_indices = np.argsort(fitnesses)[:population_size]
        population = [population[i] for i in best_indices]

        best_fitness = fitnesses[best_indices[0]]
        print(f"Geração {generation}: Melhor Fitness = {best_fitness}")

    best_index = np.argmin([gtsp.evaluate(ind) for ind in population])
    return population[best_index]


def load_tsplib_instance(filename):
    """Carrega instância da TSPLIB."""
    problem = tsplib95.load(filename)
    nodes = list(problem.get_nodes())
    cost_matrix = np.zeros((len(nodes), len(nodes)))
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i != j:
                cost_matrix[i][j] = problem.get_weight(i + 1, j + 1)
            else:
                cost_matrix[i][j] = np.inf
    return nodes, cost_matrix


def create_clusters(nodes, num_clusters):
    """Cria clusters manualmente."""
    random.shuffle(nodes)
    clusters = []
    cluster_size = len(nodes) // num_clusters
    for i in range(0, len(nodes), cluster_size):
        clusters.append(nodes[i : i + cluster_size])
    return clusters


# Uso com instância de exemplo
filename = "pcb442.tsp"
nodes, cost_matrix = load_tsplib_instance(filename)
num_clusters = 89
clusters = create_clusters(nodes, num_clusters)

gtsp = GTSP(clusters, cost_matrix)
best_solution = genetic_algorithm(
    gtsp, population_size=5 * num_clusters, generations=1000, mutation_rate=0.05
)
best_solution_val = gtsp.evaluate(best_solution)

print("Melhor Solução: ", best_solution)
print("Valor total: ", best_solution_val)
