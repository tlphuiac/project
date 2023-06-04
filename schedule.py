def calculate_travel_time(distance, TRAVEL_SPEED):
    return distance / TRAVEL_SPEED

def calculate_schedules(routes, distance_matrix, customers, depots, travel_speed):
    schedules = []

    for route in routes:
        schedule = []
        current_time = 0

        for i in range(len(route) - 1):
            source = route[i]
            destination = route[i + 1]

            if source < len(customers):
                source_x = customers[source]['x']
                source_y = customers[source]['y']
            else:
                source_x = depots[source - len(customers)]['x']
                source_y = depots[source - len(customers)]['y']

            if destination < len(customers):
                destination_x = customers[destination]['x']
                destination_y = customers[destination]['y']
            else:
                destination_x = depots[destination - len(customers)]['x']
                destination_y = depots[destination - len(customers)]['y']

            distance = distance_matrix[source][destination]
            travel_time = calculate_travel_time(distance, travel_speed)
            current_time += travel_time

            schedule.append({
                'Source': source,
                'Destination': destination,
                'Distance': distance,
                'Travel Time': travel_time,
                'Arrival Time': current_time,
                'Source X': source_x,
                'Source Y': source_y,
                'Destination X': destination_x,
                'Destination Y': destination_y
            })

        schedules.append(schedule)

    return schedules
