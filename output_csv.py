import csv

# Function to export routes to a CSV file
def export_routes(routes, instance_file):
    filename = instance_file.replace('.txt', '_routes.csv')
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Vehicle', 'Route'])
        for i, route in enumerate(routes):
            writer.writerow([f'Vehicle {i+1}', ' -> '.join(map(str, route))])

# Function to export schedules to a CSV file
def export_schedules(schedules, instance_file):
    filename = instance_file.replace('.txt', '_schedules.csv')
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Vehicle', 'Customer', 'Arrival Time', 'Start Service Time', 'End Service Time'])
        for i, schedule in enumerate(schedules):
            for entry in schedule:
                writer.writerow([f'Vehicle {i+1}', entry['Customer'], entry['Arrival Time'], entry['Start Service Time'], entry['End Service Time']])

# Function to export total distance to a CSV file
def export_total_distance(total_distance, instance_file):
    filename = instance_file.replace('.txt', '_total_distance.csv')
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Vehicle', 'Total Distance'])
        writer.writerow(['All', total_distance])