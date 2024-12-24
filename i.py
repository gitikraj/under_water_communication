import matplotlib.pyplot as plt
import numpy as np
import random
import math
from scipy.special import lambertw, erfcinv

# Constants
c_lamda = 0.3
r_dc = 1
r_bg = 1
T = 10**-9
P_e = 10**-9
pi = math.pi
h = 6.626e-34  # Planck's constant
c = 3e8  # Speed of light
R_b = 1e9  # Bit rate
neta = 0.3
lamda = 530e-9  # Wavelength in meters
P_t = 100
neta_r = 0.9
neta_t = 0.9
A_r = 1.7 * 10**-4

# Transmission range calculation
def calculate_transmission_range(theta):
    R = (2 * math.cos(theta) / c_lamda) * lambertw(
        c_lamda
        / (
            (2 * math.cos(theta))
            * math.sqrt(
                (
                    (2 * pi * (1 - math.cos(theta)) * T * h * c * R_b)
                    / (neta * lamda * P_t * neta_r * neta_t * A_r * math.cos(theta))
                )
                * (
                    (
                        math.sqrt(r_dc + r_bg)
                        + (math.sqrt(2 / T)) * erfcinv(2 * P_e)
                    )
                    ** 2
                    - r_dc
                    - r_bg
                )
            )
        )
    ).real
    return R

# Distance between two nodes
def calculate_distance(node1, node2):
    return math.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)

# Angle sum calculation
def calculate_angle_sum(node_i, node_j, destination):
    def calculate_angle(point1, point2):
        return math.atan2(point2[1] - point1[1], point2[0] - point1[0])
    
    angle1 = calculate_angle(node_i, node_j)
    angle2 = calculate_angle(node_j, destination)
    
    # Normalize angular difference to [0, π]
    angle_sum = abs(angle2 - angle1) % (2 * math.pi)
    if angle_sum > math.pi:
        angle_sum = 2 * math.pi - angle_sum
    
    return angle_sum

# Residual energy
def calculate_residual_energy():
    return random.uniform(50, 100)

# Maximum weight calculation
def calculate_max_weight(epsilon, residual_energy, angle_sum):
    return epsilon * residual_energy + (1 - epsilon) * math.sin(angle_sum)

# TDAD Algorithm
def tdad_algorithm(topology, source_id, target_id):
    # Initialize
    route = []
    temp_id = source_id
    N = len(topology)

    coordinates = topology
    epsilon = 0.5  # Weight factor

    while temp_id != target_id:
        I = []  # List of primary possible nodes
        
        # Calculate transmission range for the current node
        theta = math.radians(60)  # Example angle in radians
        dis_min = calculate_transmission_range(theta)
        
        # Check if the target is within transmission range
        distance_to_target = calculate_distance(coordinates[temp_id], coordinates[target_id])
        if distance_to_target <= dis_min:
            route.append(target_id)
            print(f"Target node {target_id} is within transmission range of node {temp_id}.")
            break
        
        # Find primary candidates
        for i in range(N):
            if i != temp_id:  # Avoid the current node
                dis = calculate_distance(coordinates[temp_id], coordinates[i])
                if dis <= dis_min:
                    I.append(i)
        
        # Calculate weights for primary candidates
        max_weight = -float("inf")
        next_node_id = None
        
        for j in I:
            angle_sum = calculate_angle_sum(coordinates[temp_id], coordinates[j], coordinates[target_id])
            residual_energy = calculate_residual_energy()
            weight = calculate_max_weight(epsilon, residual_energy, angle_sum)
            if weight > max_weight:
                max_weight = weight
                next_node_id = j
        
        if next_node_id is None:
            print("No valid next node found!")
            break

        # Update the route and move to the next node
        route.append(next_node_id)
        temp_id = next_node_id

    return route

# Plot the topology
def plot_topology(topology, route, source_id, target_id):
    # Extract coordinates
    x_coords = [coord[0] for coord in topology.values()]
    y_coords = [coord[1] for coord in topology.values()]

    # Plot all nodes
    plt.scatter(x_coords, y_coords, c="blue", label="Nodes")

    # Highlight source and target nodes
    plt.scatter(topology[source_id][0], topology[source_id][1], c="green", s=100, label="Source")
    plt.scatter(topology[target_id][0], topology[target_id][1], c="red", s=100, label="Target")

    # Annotate nodes
    for node_id, (x, y) in topology.items():
        plt.text(x + 1, y, str(node_id), fontsize=9, ha="center", color="black")

    # Draw the route
    if route:
        route_coords = [topology[source_id]] + [topology[node_id] for node_id in route]
        route_x, route_y = zip(*route_coords)
        plt.plot(route_x, route_y, c="orange", linewidth=2, label="Route")

    # Plot settings
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.title("Randomly Generated Topology with Route")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
N = 100  # Number of nodes
source_id = 0  # Source node ID
target_id = 99  # Target node ID

# Generate random topology
topology = {i: (random.randint(0, 100), random.randint(0, 100)) for i in range(N)}

# Compute the route
route = tdad_algorithm(topology, source_id, target_id)

# Plot the topology with the computed route
plot_topology(topology, route, source_id, target_id)
