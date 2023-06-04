import math

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
