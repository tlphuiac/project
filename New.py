import csv
import random
import math

#Function to calculate the travel time
def calculate_travel_time(start_node, end_node):
    # Calculate the travel time between two nodes (e.g., using the assumed velocity)
    distance = calculate_distance(start_node, end_node)
    travel_time = distance / (5 / 6)
    return travel_time

#Function to calculate the distance between two nodes
def calculate_distance(start_node, end_node):
    # Calculate the Euclidean distance between two nodes
    x1, y1 = start_node['x'], start_node['y']
    x2, y2 = end_node['x'], end_node['y']
    distance = ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5
    return distance

# Step 1: Load data
# Function to load instance data from a file
def load_instance_data(instance_file):
    with open(instance_file, 'r') as file:
        lines = file.readlines()

    num_vehicles = int(lines[0].split()[1])
    num_customers = int(lines[0].split()[2])
    num_depots = int(lines[0].split()[3])

    customers = []
    depots = []
    vehicles = []

    # Extract vehicle data
    for i in range(1, num_vehicles+1):
        vehicle_data = lines[i].split()
        vehicle = {
            'Max Duration': int(vehicle_data[0]),
            'Max Load': int(vehicle_data[1])
        }
        vehicles.append(vehicle)

    # Extract customer data
    for i in range(num_vehicles+1, num_vehicles + num_customers+1):
        customer_data = lines[i].split()
        customer = {
            'index': int(customer_data[0]),
            'x': float(customer_data[1]),
            'y': float(customer_data[2]),
            'service_time': int(customer_data[3]),
            'demand': int(customer_data[4]),
            'ready_time': int(customer_data[-2]),
            'due_time': int(customer_data[-1]),
        }
        customers.append(customer)

    # Extract depot data
    for i in range(num_customers+num_vehicles+1, num_vehicles+num_customers+num_depots+1):
        depot_data = lines[i].split()
        depot = {
            'index': int(depot_data[0]),
            'x': float(depot_data[1]),
            'y': float(depot_data[2]),
            'Max Capacity': int(depot_data[-1])
        }
        depots.append(depot)

    return num_vehicles, num_customers, num_depots, customers, depots, vehicles

def calculate_route_distance(individual):
    distance = 0
    for i in range(len(route) - 1):
        customer1 = customers[route[i]]
        customer2 = customers[route[i + 1]]
        distance += calculate_distance(customer1['x'], customer1['y'], customer2['x'], customer2['y'])
    return distance

def calculate_route_duration(route):
    duration = 0.0
    current_time = 0.0
    for i in range(len(route) - 1):
        customer1 = customers[route[i]]
        customer2 = customers[route[i + 1]]
        distance = calculate_distance(customer1['x'], customer1['y'], customer2['x'], customer2['y'])
        current_time += distance/(5/6)
        current_time = max(current_time, customer2['ready_time'])
        current_time += customer2['service_time']
        duration = max(duration, current_time - customer2['ready_time'])
    return duration

# Generate the distance matrix
def generate_distance_matrix():
    num_nodes = num_customers + num_depots
    distance_matrix = [[0] * num_nodes for _ in range(num_nodes)]

    # Calculate distances between customers
    for i in range(num_customers):
        for j in range(i+1, num_customers):
            distance = calculate_distance(customers[i], customers[j])
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance

    # Calculate distances between depots and customers
    for i in range(num_customers):
        for j in range(num_customers, num_nodes):
            distance = calculate_distance(customers[i], depots[j-num_customers])
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance

    return distance_matrix

def save_distance_matrix(distance_matrix, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in distance_matrix:
            writer.writerow(row)

# Step 2: Define parameters
population_size = 50
num_generations = 100
tournament_size = 5
crossover_rate = 0.8
mutation_rate = 0.2
penalty_weight_1 = 10000
penalty_weight_2 = 10000
penalty_weight_3 = 10000
penalty_weight_4 = 10000
penalty_weight_6 = 10000
penalty_weight_7_8 = 10000

# Step 4: Define fitness calculation
def calculate_fitness(individual):
    total_distance = 0.0
    total_duration = 0.0
    penalty = 0.0

    for vehicle_routes in individual:
        for route in vehicle_routes:
            total_distance += calculate_route_distance(route)
            total_duration += calculate_route_duration(route)

            # Constraint 1: Ensuring the vehicle that leaves the customer is the same as the one that visits the customer
            if route[0] != route[-1]:
                penalty += penalty_weight_1

            # Constraint 2: Each customer is assigned to a vehicle
            assigned_customers = [route[0]] + route + [route[-1]]
            for i in range(len(assigned_customers) - 1):
                if assigned_customers[i] == assigned_customers[i + 1]:
                    penalty += penalty_weight_2

            # Constraint 3: Determining whether each customer is assigned to a depot or not
            for i in range(1, len(route) - 1):
                customer = customers[route[i]]
                depot = depots[customer['vehicle_index']]
                if depot['index'] != route[0] and depot['index'] != route[-1]:
                    penalty += penalty_weight_3

            # Constraint 4: Maximum load and duration for each vehicle
            total_demand = 0
            total_service_time = 0
            for i in range(len(route) - 2):
                customer = customers[route[i + 1]]
                total_demand += customer['demand']
                total_service_time += customer['service_time']
                if total_demand > vehicles[route[0]]['Max Load'] or total_service_time > vehicles[route[0]]['Max Duration']:
                    penalty += penalty_weight_4

            # Constraint 6: Time constraints between consecutive customers
            current_time = 0.0
            for i in range(len(route) - 1):
                customer1 = customers[route[i]]
                customer2 = customers[route[i + 1]]
                distance = calculate_distance(customer1['x'], customer1['y'], customer2['x'], customer2['y'])
                current_time += distance
                current_time = max(current_time, customer2['ready_time'])
                current_time += customer2['service_time']
                if current_time > customer2['due_time']:
                    penalty += penalty_weight_6

            # Constraint 7 & 8: Time windows for each customer
            for i in range(1, len(route) - 1):
                customer = customers[route[i]]
                if customer['ready_time'] > current_time:
                    penalty += penalty_weight_7_8
                current_time = max(current_time, customer['ready_time'])
                current_time += customer['service_time']

    fitness = total_distance + penalty
    return fitness, total_distance, total_duration

# Step 5: Generate initial population
def generate_initial_population():
    population = []
    for _ in range(population_size):
        individual = []
        for _ in range(num_vehicles):
            vehicle_sequence = random.sample(range(num_customers), num_customers)
            individual.append(vehicle_sequence)
        population.append(individual)
    return population

# Step 6: Genetic operators (Selection, Crossover, Mutation)
def tournament_selection(population, fitness_values):
    selected = []
    for _ in range(population_size):
        tournament = random.sample(range(population_size), tournament_size)
        winner = tournament[0]
        for challenger in tournament[1:]:
            if fitness_values[challenger] < fitness_values[winner]:
                winner = challenger
        selected.append(population[winner])
    return selected

def best_cost_route_crossover(parent1, parent2):
    child = []
    for i in range(num_vehicles):
        if random.random() < crossover_rate:
            # Perform best cost route crossover
            vehicle_route1 = parent1[i]
            vehicle_route2 = parent2[i]
            common_customers = list(set(vehicle_route1) & set(vehicle_route2))
            while len(common_customers) > 0:
                best_cost_customer = None
                min_cost = float('inf')
                for customer in common_customers:
                    cost1 = calculate_route_distance(vehicle_route1 + [customer]) - calculate_route_distance(vehicle_route1)
                    cost2 = calculate_route_distance(vehicle_route2 + [customer]) - calculate_route_distance(vehicle_route2)
                    if cost1 + cost2 < min_cost:
                        min_cost = cost1 + cost2
                        best_cost_customer = customer
                vehicle_route1.append(best_cost_customer)
                vehicle_route2.append(best_cost_customer)
                common_customers.remove(best_cost_customer)
        child.append(vehicle_route1)
    return child

def swap_mutation(individual):
    mutated_individual = []
    for i in range(num_vehicles):
        if random.random() < mutation_rate:
            # Perform swap mutation
            vehicle_sequence = individual[i]
            idx1, idx2 = random.sample(range(1, num_customers), 2)
            vehicle_sequence[idx1], vehicle_sequence[idx2] = vehicle_sequence[idx2], vehicle_sequence[idx1]
        mutated_individual.append(vehicle_sequence)
    return mutated_individual

# Step 7: Evolutionary process
def evolve(population):
    new_population = []

    # Calculate fitness for each individual
    fitness_values = []
    for individual in population:
        fitness_value, _, _ = calculate_fitness(individual)
        fitness_values.append(fitness_value)

    # Selection
    selected_individuals = tournament_selection(population, fitness_values)

    # Crossover
    for i in range(0, population_size, 2):
        parent1 = selected_individuals[i]
        parent2 = selected_individuals[i + 1]
        child1 = best_cost_route_crossover(parent1, parent2)
        child2 = best_cost_route_crossover(parent2, parent1)
        new_population.append(child1)
        new_population.append(child2)

    # Mutation
    for individual in new_population:
        mutated_individual = swap_mutation(individual)
        new_population.append(mutated_individual)

    return new_population

# Step 8: Solve MDVRPTW using Genetic Algorithm
def genetic_algorithm(file_path, population_size, num_generations, tournament_size, mutation_rate):
    # Step 1: Preprocessing
    num_vehicles, num_customers, num_depots, customers, depots, vehicles = load_instance_data(instance_file)

    # Step 2: Initialization
    population = generate_initial_population()



#distance_matrix = generate_distance_matrix()
#save_distance_matrix(distance_matrix, 'distance_matrix.csv')

# Print a message to confirm the file has been saved
#print("Distance matrix has been saved as 'distance_matrix.csv'.")
#individual = generate_random_individual()
instance_file = 'D:\\Thesis\\Project_thesis\\pr11a.txt'
num_vehicles, num_customers, num_depots, customers, depots, vehicles = load_instance_data(instance_file)
population = generate_initial_population()
print()

#best_solution = genetic_algorithm(population_size, num_generations, tournament_size, mutation_rate)












