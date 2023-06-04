# Function to load instance data from a file
import csv

def load_instance_data(instance_file):
    with open(instance_file, 'r') as file:
        lines = file.readlines()

    num_vehicles = int(lines[0].split()[1])
    num_customers = int(lines[0].split()[2])
    num_depots = int(lines[0].split()[3])

    customers = []
    depots = []

    # Extract customer data
    for i in range(5, num_customers + 5):
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
    for i in range(-1, -4):
        depot_data = lines[i].split()
        depot = {
            'index': int(depot_data[0]),
            'x': float(depot_data[1]),
            'y': float(depot_data[2]),
            'Max Capacity': int(depot_data[-1])
        }
        depots.append(depot)

    return num_vehicles, num_customers, num_depots, customers, depots


