import csv
import random
import math


# Constants
population_size = 100
MAX_GENERATIONS = 100
MUTATION_RATE = 0.1
VELOCITY = 5/6

# Function to calculate the travel time based on distance and velocity
# def calculate_travel_time(distance):
#    time = distance / VELOCITY
#   return time

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


def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def create_distance_matrix(customers, depots):
    num_customers = len(customers)
    num_depots = len(depots)
    num_nodes = num_customers + num_depots

    distance_matrix = [[0] * num_nodes for _ in range(num_nodes)]

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i < num_customers:
                node_i = customers[i]
            else:
                node_i = depots[i - num_customers]

            if j < num_customers:
                node_j = customers[j]
            else:
                node_j = depots[j - num_customers]

            distance = calculate_distance(node_i['x'], node_i['y'], node_j['x'], node_j['y'])
            distance_matrix[i][j] = distance

    return distance_matrix

# Step 2: Initialization
def initialize_population(num_vehicles, num_customers):
    population = []
    for _ in range(population_size):
        solution = [[] for _ in range(num_vehicles)]
        customers = list(range(1, num_customers + 1))
        random.shuffle(customers)

        for customer in customers:
            vehicle = random.randint(0, num_vehicles - 1)
            solution[vehicle].append(customer)

        population.append(solution)

    return population


# Step 3: Fitness Evaluation
def evaluate_fitness(individual, customers, depots, distance_matrix):
    total_distance = 0
    total_time = 0
    total_load = 0
    last_customer_index = 0
    routes = []
    schedules = []
    for i, chromosome in enumerate(individual):
        vehicle_distance = 0
        vehicle_time = 0
        vehicle_load = 0
        for j in range(num_depots):
            depot = depots[j]
        return depot
        last_customer = depot
        route = [depot['index']]
        schedule = []
        for customer_index in chromosome:
            customer = customers[customer_index - 1]
            distance = distance_matrix[last_customer['index'] - 1][customer['index'] - 1]
            travel_time = calculate_travel_time(distance)
            if hasattr(last_customer,'due_time'):
                arrival_time = max(last_customer['due_time'], customer['ready_time']) + travel_time
            else:
                arrival_time = customer['ready_time'] + travel_time
            wait_time = max(0, customer['ready_time'] - arrival_time)
            start_service_time = arrival_time + wait_time
            end_service_time = start_service_time + customer['service_time']

            route.append(customer['index'])
            schedule.append({
                'Customer': customer['index'],
                'Arrival Time': arrival_time,
                'Start Service Time': start_service_time,
                'End Service Time': end_service_time
            })
            vehicle_distance += distance
            vehicle_time += travel_time + wait_time + customer['service_time']
            vehicle_load += customer['demand']
            last_customer = customer
        distance_back_to_depot = distance_matrix[last_customer['index'] - 1][depot['index'] - 1]
        vehicle_distance += distance_back_to_depot
        vehicle_time += calculate_travel_time(distance_back_to_depot)
        route.append(depot['index'])
        routes.append(route)
        schedules.append(schedule)
        total_distance += vehicle_distance
        total_time += vehicle_time
        total_load += vehicle_load

    fitness = 1 / total_distance if total_demand <= depots[0]['Max Capacity'] else 0

    return fitness, routes, schedules, total_distance

# Step 4: Tournament Selection
def tournament_selection(population, tournament_size):
    selected_parents = []
    for _ in range(len(population)):
        tournament = random.sample(population, tournament_size)
        best_solution = min(tournament, key=lambda x: evaluate_fitness(x))
        selected_parents.append(best_solution)
    return selected_parents

# Step 5: Best Cost Route Crossover
def best_cost_route_crossover(parent1, parent2):
    offspring1 = []
    offspring2 = []

    for i in range(len(parent1)):
        route1 = parent1[i]
        route2 = parent2[i]
        combined_route = route1 + route2

        while combined_route:
            best_customer = None
            best_distance = float('inf')

            for customer in combined_route:
                route1_distance = calculate_distance(offspring1[-1], customer) if offspring1 else 0
                route2_distance = calculate_distance(offspring2[-1], customer) if offspring2 else 0

                if route1_distance < best_distance:
                    best_customer = customer
                    best_distance = route1_distance

                if route2_distance < best_distance:
                    best_customer = customer
                    best_distance = route2_distance

            if best_customer in route1:
                route1.remove(best_customer)
                offspring1.append(best_customer)
            else:
                route2.remove(best_customer)
                offspring2.append(best_customer)

        offspring1.append(0)
        offspring2.append(0)

    return offspring1, offspring2

# Step 6: Swap Mutation
def swap_mutation(solution, mutation_rate):
    mutated_solution = solution.copy()
    for i in range(len(mutated_solution)):
        if random.random() < mutation_rate:
            route = mutated_solution[i]
            if len(route) > 2:
                idx1, idx2 = random.sample(range(1, len(route) - 1), 2)
                route[idx1], route[idx2] = route[idx2], route[idx1]
    return mutated_solution

# Step 7: Population Update
def create_next_generation(population, tournament_size, mutation_rate):
    selected_parents = tournament_selection(population, tournament_size)
    offspring = []

    for i in range(0, len(selected_parents), 2):
        parent1 = selected_parents[i]
        parent2 = selected_parents[i+1]
        child1, child2 = best_cost_route_crossover(parent1, parent2)
        offspring.extend([child1, child2])

    next_generation = offspring.copy()

    for solution in next_generation:
        mutated_solution = swap_mutation(solution, mutation_rate)
        next_generation.append(mutated_solution)

    return next_generation

# Step 8: Genetic Algorithm
def genetic_algorithm(file_path, population_size, num_generations, tournament_size, mutation_rate):
    # Step 1: Preprocessing
    num_vehicles, num_customers, num_depots, customers, depots, vehicles = load_instance_data('D:\\Thesis\\Project_thesis\\pr11a.txt')

    # Step 2: Initialization
    population = initialize_population(num_vehicles, num_customers)

    # Initialize variables to track the best solution
    best_fitness = -1
    best_solution = None

    # Step 3: Evaluate the fitness of the initial population
    fitness_values = []
    distance_matrix=create_distance_matrix(customers, depots)
    for individual in population:
        fitness, _, _, _ = evaluate_fitness(individual, customers, depots, distance_matrix)
        fitness_values.append(fitness)

        # Update the best solution if necessary
        if fitness > best_fitness:
            best_fitness = fitness
            best_solution = individual
    # Step 4: Evolution loop
    for generation in range(num_generations):
        # Step 5: Selection
        selected_individuals = tournament_selection(population, fitness_values, tournament_size)
        # Step 6: Crossover
        offspring_population = []
        for i in range(0, population_size, 2):
            parent1 = selected_individuals[i]
            parent2 = selected_individuals[i + 1]
            offspring1, offspring2 = best_cost_route_crossover(parent1, parent2)
            offspring_population.extend([offspring1, offspring2])
            # Step 7: Mutation
            mutated_population = []
            for individual in offspring_population:
                mutated_individual = swap_mutation(individual, mutation_rate)
                mutated_population.append(mutated_individual)

        # Update the population with the mutated population
        population = mutated_population

# Find the best solution in the current generation
        best_index = fitness_values.index(min(fitness_values))
        if fitness_values[best_index] < best_fitness:
            best_fitness = fitness_values[best_index]
            best_solution = population[best_index]

        print(f"Generation {generation + 1} - Best Fitness: {best_fitness}")

    # Export the results to files
    export_routes(routes, instance_file)
    export_schedules(schedules, instance_file)
    export_total_distance(total_distance, instance_file)

    return routes, schedules, total_distance

# Function to export the routes to a CSV file
def export_routes(routes, instance_file):
    with open(f'{instance_file}_routes.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Vehicle', 'Route'])
        for i, route in enumerate(routes, start=1):
            writer.writerow([f'Vehicle {i}', ' -> '.join(map(str, route))])

# Function to export the schedules to a CSV file
def export_schedules(schedules, instance_file):
    with open(f'{instance_file}_schedules.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Vehicle', 'Customer', 'Arrival Time', 'Start Service Time', 'End Service Time'])
        for i, schedule in enumerate(schedules, start=1):
            for item in schedule:
                writer.writerow([f'Vehicle {i}', item['Customer'], item['Arrival Time'], item['Start Service Time'], item['End Service Time']])

# Function to export the total distance traveled by each vehicle to a CSV file
def export_total_distance(total_distance, instance_file):
    with open(f'{instance_file}_total_distance.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Vehicle', 'Total Distance'])
        for i, distance in enumerate(total_distance, start=1):
            writer.writerow([f'Vehicle {i}', distance])

# Run the genetic algorithm and export the results
#routes, schedules, total_distance = genetic_algorithm(r'D:\Thesis\Project_thesis\pr11a.txt')
# Call the genetic_algorithm function with your desired parameters
#genetic_algorithm('D:\\Thesis\\Project_thesis\\pr11a.txt', population_size=100, num_generations=100, tournament_size=5, mutation_rate=0.1)
#genetic_algorithm('D:\\Thesis\\Project_thesis\\pr11a.txt', 100, 100, 5, 0.1)
file_path = ('D:\\Thesis\\Project_thesis\\pr11a.txt')
population_size = 100
num_generations = 100
tournament_size = 5
mutation_rate = 0.1
# Step 1: Preprocessing
num_vehicles, num_customers, num_depots, customers, depots, vehicles = load_instance_data('D:\\Thesis\\Project_thesis\\pr11a.txt')



print(depots)
