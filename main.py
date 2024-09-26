import tsplib95
import random
import numpy as np


class GTSP:
    def __init__(self, clusters, cost_matrix):
        """
        clusters: lista de clusters, cada cluster é uma lista de nós.
        cost_matrix: matriz de custo onde cost_matrix[i][j] indica o custo de ir do nó i para o nó j.
        """
        self.clusters = clusters
        self.cost_matrix = cost_matrix
        self.num_clusters = len(clusters)
        self.nodes = [node for cluster in clusters for node in cluster]

    def evaluate(self, individual):
        """Avalia o custo total do tour dado um indivíduo (C, N)."""
        cluster_order, node_selection = individual
        tour_cost = 0
        node_selection = [node - 1 for node in node_selection]
        # Calcular o custo total do ciclo
        for i in range(self.num_clusters - 1):
            node_a = node_selection[cluster_order[i]]
            node_b = node_selection[cluster_order[i + 1]]
            tour_cost += self.cost_matrix[node_a][node_b]

        # Adicionar o custo de voltar ao início
        first_node = node_selection[cluster_order[0]]
        last_node = node_selection[cluster_order[-1]]
        tour_cost += self.cost_matrix[last_node][first_node]
        return tour_cost


# Operadores Genéticos
def crossover(parent1, parent2):
    """Operador de crossover baseado no artigo."""
    cut_point = random.randint(
        1, len(parent1[0]) - 2
    )  # Cortar no meio para trocar clusters
    child1_c = parent1[0][:cut_point] + [
        c for c in parent2[0] if c not in parent1[0][:cut_point]
    ]
    child2_c = parent2[0][:cut_point] + [
        c for c in parent1[0] if c not in parent2[0][:cut_point]
    ]

    # Realizar crossover na sequência de nós (N)
    child1_n = parent1[1][:]
    child2_n = parent2[1][:]

    return (child1_c, child1_n), (child2_c, child2_n)


def intra_cluster_mutation(individual, clusters):
    """Mutação intra-cluster: troca o nó selecionado dentro de um cluster."""
    cluster_order, node_selection = individual
    mutated_node_selection = node_selection[:]
    cluster_idx = random.randint(0, len(clusters) - 1)  # Seleciona um cluster aleatório
    mutated_node_selection[cluster_idx] = random.choice(
        clusters[cluster_idx]
    )  # Troca nó no cluster
    return (cluster_order, mutated_node_selection)


def inter_cluster_mutation(individual):
    """Mutação inter-cluster: troca dois clusters de lugar."""
    cluster_order, node_selection = individual
    mutated_cluster_order = cluster_order[:]
    i, j = random.sample(range(len(cluster_order)), 2)
    mutated_cluster_order[i], mutated_cluster_order[j] = (
        mutated_cluster_order[j],
        mutated_cluster_order[i],
    )
    return (mutated_cluster_order, node_selection)


def selection(population, fitnesses, num_to_select):
    """Seleção baseada no método de torneio binário."""
    selected_indices = []
    for _ in range(num_to_select):
        i, j = random.sample(range(len(population)), 2)
        selected_indices.append(i if fitnesses[i] < fitnesses[j] else j)
    return [population[i] for i in selected_indices]


# Algoritmo Genético Principal
def genetic_algorithm(gtsp, population_size=50, generations=100, mutation_rate=0.05):
    # Inicializa a população
    population = []
    for _ in range(population_size):
        cluster_order = random.sample(range(gtsp.num_clusters), gtsp.num_clusters)
        node_selection = [random.choice(cluster) for cluster in gtsp.clusters]
        population.append((cluster_order, node_selection))

    for generation in range(generations):
        fitnesses = [gtsp.evaluate(ind) for ind in population]  # Avaliação
        num_parents = population_size // 2
        parents = selection(population, fitnesses, num_parents)  # Seleção de pais

        # Geração de filhos via crossover
        offspring = []
        for i in range(0, len(parents) - 1, 2):
            child1, child2 = crossover(parents[i], parents[i + 1])
            offspring.append(child1)
            offspring.append(child2)

        # Aplicar mutação nos filhos
        for i in range(len(offspring)):
            if random.random() < mutation_rate:
                offspring[i] = intra_cluster_mutation(
                    offspring[i], gtsp.clusters
                )  # Mutação intra-cluster
            if random.random() < mutation_rate:
                offspring[i] = inter_cluster_mutation(
                    offspring[i]
                )  # Mutação inter-cluster

        population += offspring  # Combina população atual com filhos
        fitnesses += [gtsp.evaluate(ind) for ind in offspring]
        best_indices = np.argsort(fitnesses)[:population_size]  # Seleção dos melhores
        population = [population[i] for i in best_indices]

        # Relatório de progresso
        best_fitness = fitnesses[best_indices[0]]
        # print(f"Geração {generation}: Melhor Fitness = {best_fitness}")

    # Retorna o melhor indivíduo encontrado
    best_index = np.argmin([gtsp.evaluate(ind) for ind in population])
    return population[best_index]


def load_tsplib_instance(filename):
    """Carrega uma instância .tsp da TSPLIB usando a biblioteca tsplib95."""
    problem = tsplib95.load(filename)
    nodes = list(problem.get_nodes())

    # Construir matriz de custos com base nas distâncias entre os nós
    cost_matrix = np.zeros((len(nodes), len(nodes)))
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i != j:
                cost_matrix[i][j] = problem.get_weight(
                    i + 1, j + 1
                )  # Nós começam em 1 no TSPLIB
            else:
                cost_matrix[i][
                    j
                ] = np.inf  # Distância de um nó para ele mesmo é infinita

    return nodes, cost_matrix


def create_clusters(nodes, num_clusters):
    """
    Cria clusters manualmente a partir dos nós carregados.
    Cada cluster terá aproximadamente `len(nodes) // num_clusters` nós.
    """
    random.shuffle(nodes)
    clusters = []
    cluster_size = len(nodes) // num_clusters
    for i in range(0, len(nodes), cluster_size):
        clusters.append(nodes[i : i + cluster_size])

    return clusters


# Exemplo de Uso com um Problema Simples (para demonstração)
filename = "d198.tsp"
nodes, cost_matrix = load_tsplib_instance(filename)
num_clusters = 40  # Defina o número de clusters desejado
clusters = create_clusters(nodes, num_clusters)
print("Número de nós:", len(nodes))
print("Dimensões da matriz de custos:", cost_matrix.shape)

# Matriz de custo aleatória para teste (deve ser substituída por instâncias TSPLIB para experimentos reais)
gtsp = GTSP(clusters, cost_matrix)
solutions = []
solutions_val = []
for i in range(50):
    print("Solução ", i + 1)
    best_solution = genetic_algorithm(gtsp)
    solutions.append(best_solution)
    best_solution_val = gtsp.evaluate(best_solution)
    solutions_val.append(best_solution_val)
min_value, min_index = min((value, index) for index, value in enumerate(solutions_val))
print("Melhor Solução: ", solutions[min_index])
print("Valor total: ", min_value)
