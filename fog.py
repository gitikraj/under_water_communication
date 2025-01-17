import numpy as np
import random
import math
from scipy.special import lambertw
from scipy.special import erfinv
import matplotlib.pyplot as plt
# import numpy as np
# import random
# import math
# from scipy.special import lambertw, erfcinv


theta = math.radians(0)  # i don't know how to define it so i take it as 0 for simplicity
theta_0 = math.radians(60)
c_lamda = 0.3
r_dc = 1
r_bg = 1
T = 10**-9
P_e = 10**-9
h = 6.62607015 * 10**-34
c = 3 * 10**-8
R_b = random.uniform(1e6, 1e8)  # Random data rate between 1 Mbps and 100 Mbps
neta = 0.3
lamda = 530
P_t = 100
neta_r = 0.9
neta_t = 0.9
A_r = 1.7 * 10**-4


def calculate_transmission_range():
    factor = 2 * math.cos(theta) / c_lamda
    small_value = 1e-9  # Small positive value to avoid division by zero or negative sqrt
    
    term1 = (2 * math.pi * (1 - math.cos(theta_0)) * T * h * c * R_b) / (neta * lamda * P_t * neta_r * neta_t * A_r * math.cos(theta))
    
    erfinv_term = erfinv(min(2 * P_e, 1 - small_value))  # Ensure erfinv input is in [-1, 1]
    sqrt_term = math.sqrt(r_dc + r_bg) + math.sqrt(2 / T) * erfinv_term
    
    term2 = (sqrt_term ** 2) - r_dc - r_bg
    term2 = max(term2, small_value)  # Ensure non-negative sqrt input

    argument = 1 / (factor * math.sqrt(term1 * term2))
    argument = max(argument, small_value)  # Prevent negative Lambert argument

    R = factor * lambertw(argument).real  # Use real part of Lambert function
    return R


def calculate_distance(node1, node2):
  return math.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)

def calculate_angle(point1, point2):
    return math.atan2(point2[1] - point1[1], point2[0] - point1[0])

def calculate_angle_sum(node_i, node_j, destination):
  """Calculate the angle sum for routing."""
  angle1 = calculate_angle(node_i, node_j)
  angle2 = calculate_angle(node_j, destination)
  deviation = abs(angle2 - angle1)
  return deviation

def calculate_residual_energy():
  residual_energy = random.uniform(50, 100)  # Example residual energy value
  return residual_energy

def calculate_max_weight(epsilon, residual_energy, angle_sum):
  max_weight = epsilon * residual_energy + (1 - epsilon) * math.sin(angle_sum)
  return max_weight

# Main function implementing the TDAD algorithm
def tdad_algorithm(topology, source_id, target_id):
  # Initialize
  route = []
  temp_id = source_id
  N = len(topology)

  coordinates = {i: (random.uniform(0, 100), random.uniform(0, 100)) for i in range(N)}

  bandwidth = {i: random.uniform(10, 100) for i in range(N)}

  energy = {i: random.uniform(10, 100) for i in range(N)}

  # Main loop
  while temp_id != target_id:
    epsilon = 0.5  # Weight factor
    I = []  # List of next possible nodes
    I_prime = []  # List of secondary possible nodes

    for i in range(N):
      dis_min = calculate_transmission_range()
      dis = calculate_distance(coordinates[temp_id], coordinates[i])
      if dis <= dis_min:
        I.append(i)

      for j in I:
        for k in range(N):
          dis = calculate_distance(coordinates[k], coordinates[j])
          if dis <= dis_min:
            I_prime.append(k)

        angle_sum = calculate_angle_sum(coordinates[k], coordinates[j], coordinates[target_id])
        residual_energy = calculate_residual_energy()
        max_weight = calculate_max_weight(epsilon, residual_energy, angle_sum)
        next_node_id = j
        temp_id = next_node_id

      route.append(temp_id)

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
