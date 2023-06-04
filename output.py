def export_results(routes, schedules, total_distance, output_file):
    with open(output_file, 'w') as file:
        # Export the routes
        file.write("Routes:\n")
        for vehicle, route in enumerate(routes):
            file.write(f"Vehicle {vehicle + 1}: {route}\n")

        file.write("\n")

        # Export the schedules
        file.write("Schedules:\n")
        for vehicle, schedule in enumerate(schedules):
            file.write(f"Vehicle {vehicle + 1}: {schedule}\n")

        file.write("\n")

        # Export the total distance
        file.write("Total Distance:\n")
        for vehicle, distance in enumerate(total_distance):
            file.write(f"Vehicle {vehicle + 1}: {distance}\n")
