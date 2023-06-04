from data import load_instance_data
from distance import *
import schedule
import genetic_algorithm
import output_csv



# Instance file
instance_file = "D:\\Thesis\\Project_thesis\\pr11a.txt"

# Load instance data
num_vehicles, num_customers, num_depots, customers, depots = load_instance_data(instance_file)

# Create distance matrix
distance_matrix = create_distance_matrix(customers, depots)

# Execute the genetic algorithm
routes, schedules, total_distance = genetic_algorithm.genetic_algorithm(instance_file, num_vehicles, num_customers, distance_matrix)

# Calculate schedules
schedules = schedule.calculate_schedules(routes, distance_matrix, customers, depots, TRAVEL_SPEED)

# Export the results to CSV files
output_csv.export_routes(routes, instance_file)
output_csv.export_schedules(schedules, instance_file)
output_csv.export_total_distance(total_distance, instance_file)

if __name__ == "__main__":
    main()