import random
import math
from data import load_instance_data
from distance import calculate_distance
from schedule import calculate_travel_time
import output_csv

# Constants
POPULATION_SIZE = 100
MAX_GENERATIONS = 100
MUTATION_RATE = 0.1
TRAVEL_SPEED = 5/6

# Function to initialize the population
def initialize_population(num_vehicles, num_customers):
    population = []

    for _ in range(POPULATION_SIZE):
        chromosome = random.sample(range(1, num_customers + 1), num_customers)
        chromosomes = [chromosome[i:i + num_vehicles] for i in range(0, len(chromosome), num_vehicles)]
        population.append(chromosomes)

    return population

# Function to calculate the fitness value of an individual
def calculate_fitness(individual, customers, depots, distance_matrix):
    total_distance = 0
    total_time = 0
    total_demand = 0
    last_customer_index = 0
    routes = []
    schedules = []
    for i, chromosome in enumerate(individual):
        vehicle_distance = 0
        vehicle_time = 0
        vehicle_demand = 0
        depot = depots[i]
        last_customer = depot
        route = [depot['index']]
        schedule = []
        for customer_index in chromosome:
            customer = customers[customer_index - 1]
            distance = distance_matrix[last_customer['index'] - 1][customer['index'] - 1]
            travel_time = calculate_travel_time(distance)
            arrival_time = max(last_customer['due_time'], customer['ready_time']) + travel_time
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
            vehicle_demand += customer['demand']
            last_customer = customer
        distance_back_to_depot = distance_matrix[last_customer['index'] - 1][depot['index'] - 1]
        vehicle_distance += distance_back_to_depot
        vehicle_time += calculate_travel_time(distance_back_to_depot)
        route.append(depot['index'])
        routes.append(route)
        schedules.append(schedule)
        total_distance += vehicle_distance
        total_time += vehicle_time
        total_demand += vehicle_demand

    if total_demand <= depots[0]['Max Capacity']:
        fitness = 1 / total_distance
    else:
        return None, None, None, None

    return fitness, routes, schedules, total_distance

# Function for tournament selection
def tournament_selection(population, customers, depots, distance_matrix):
    tournament_size = int(POPULATION_SIZE * 0.1)
    tournament_candidates = random.sample(population, tournament_size)
    best_candidate = max(tournament_candidates, key=lambda x: calculate_fitness(x, customers, depots, distance_matrix))
    return best_candidate

# Function for ordered crossover
def ordered_crossover(parent1, parent2):
    size = len(parent1)
    child = [-1] * size
    start = random.randint(0, size - 1)
    end = random.randint(start + 1, size)
    for i in range(start, end):
        child[i] = parent1[i]
    remaining = [gene for gene in parent2 if gene not in child]
    j = 0
    for i in range(size):
        if child[i] == -1:
            child[i] = remaining[j]
            j += 1
    return child

# Function for mutation (exchange mutation)
def mutation(individual):
    size = len(individual)
    index1 = random.randint(0, size - 1)
    index2 = random.randint(0, size - 1)
    individual[index1], individual[index2] = individual[index2], individual[index1]
    return individual

# Main genetic algorithm function
def genetic_algorithm(instance_file, num_vehicles, num_customers, distance_matrix):
    # Load instance data
    num_vehicles, num_customers, num_depots, customers, depots = load_instance_data(instance_file)

    # Initialize the population
    population = initialize_population(num_vehicles, num_customers)

    best_fitness = 0
    best_solution = None

    # Run the genetic algorithm for a fixed number of generations
    for generation in range(MAX_GENERATIONS):
        new_population = []

        # Elitism: Select the best individual from the previous generation
        best_individual = max(population, key=lambda x: calculate_fitness(x, customers, depots, distance_matrix)[0])
        best_fitness = calculate_fitness(best_individual, customers, depots, distance_matrix)[0]
        best_solution = best_individual.copy()
        new_population.append(best_individual)

        # Generate new individuals through selection, crossover, and mutation
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, customers, depots, distance_matrix)
            parent2 = tournament_selection(population, customers, depots, distance_matrix)

            child = ordered_crossover(parent1, parent2)

            if random.random() < MUTATION_RATE:
                child = mutation(child)

            new_population.append(child)

        population = new_population

    # Calculate the fitness and extract the routes, schedules, and total distance of the best solution
    fitness, routes, schedules, total_distance = calculate_fitness(best_solution, customers, depots, distance_matrix)

    # Export the results to CSV files
    output_csv.export_routes(routes, instance_file)
    output_csv.export_schedules(schedules, instance_file)
    output_csv.export_total_distance(total_distance, instance_file)

    return routes, schedules, total_distance


