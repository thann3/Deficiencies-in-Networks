#Simple demonstration of an arbitrary node network
#All nodes are weighed the same
#Edge loss simulation

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Network structure (node connections)
network = {
    1: [2, 3],
    2: [1, 3],
    3: [1, 2, 5],
    4: [5],
    5: [3, 4]
}

# Number of nodes in the network
num_nodes = len(network)

# c_values for each client (c1, c2, c3, c4, c5)
c_values = [1, 2, 3, 4, 5]

# Mixing step: Nodes send their estimates to neighbors and receive estimates from neighbors
def mixing_step(estimates, disconnected_edge):
    new_estimates = np.zeros(num_nodes)
    for node, neighbors in network.items():
        if disconnected_edge is not None and node in disconnected_edge:
            # If the current node is disconnected, perform mixing update with connected neighbors only
            connected_neighbors = [neighbor for neighbor in neighbors if neighbor not in disconnected_edge]
            received_updates = [estimates[neighbor - 1] for neighbor in connected_neighbors]
        else:
            received_updates = [estimates[neighbor - 1] for neighbor in neighbors]
        new_estimates[node - 1] = np.mean(received_updates)
    return new_estimates

# Gradient step with partial aggregation: Perform a gradient update on each connected node's estimate
def gradient_step(estimates, disconnected_edge, learning_rate=0.1):
    new_estimates = estimates.copy()
    for node, neighbors in network.items():
        # Only perform the gradient update if the node is still connected
        if disconnected_edge is None or (node not in disconnected_edge and node not in disconnected_edge[::-1]):
            gradient = 2 * (estimates[node - 1] - c_values[node - 1])
            new_estimates[node - 1] -= learning_rate * gradient
        else:
            # When disconnected, update nodes 4 and 5 using their correct target values
            gradient = 2 * (estimates[node - 1] - c_values[node - 1])
            new_estimates[node - 1] -= learning_rate * gradient
    return new_estimates

def asynchronous_decentralized_gradient_descent(num_epochs, learning_rate, edge_disconnect_prob):
    # Initialize estimates randomly for each node within the range [0, 5]
    estimates = np.random.uniform(0, 5, num_nodes)

    all_estimates = [estimates.copy()]

    for epoch in range(num_epochs):
        # Decide whether to disconnect the edge based on the given probability
        disconnected_edge = (1, 2) if np.random.random() < edge_disconnect_prob else None

        # Mixing step: Nodes exchange their estimates, excluding the disconnected edge if provided
        estimates = mixing_step(estimates, disconnected_edge)

        # Gradient step with partial aggregation: Nodes perform a gradient update
        estimates = gradient_step(estimates, disconnected_edge, learning_rate)

        all_estimates.append(estimates.copy())

    return all_estimates

# Parameters
num_epochs = 10000
learning_rate = 0.01
edge_disconnect_prob = 1 #Set the probability of edge disconnection here (0.5 for 50%)

# Print initial estimated values for clients 1, 2, 3, 4, and 5
initial_estimates = np.random.uniform(0, 5, num_nodes)
for node in range(1, num_nodes + 1):
    initial_estimate = initial_estimates[node - 1]
    print(f"Initial x{node}: {initial_estimate:.2f}")

# Run the asynchronous decentralized gradient descent algorithm with the specified edge disconnect probability
all_estimates = asynchronous_decentralized_gradient_descent(num_epochs, learning_rate, edge_disconnect_prob)

# Calculate the optimal solution x*
x_star = np.mean(c_values)
print(f"x*: {x_star:.2f}")

L1_norm = 0
# Print final estimated values for clients 1, 2, 3, 4, and 5
for node in range(1, num_nodes + 1):
    final_estimate = all_estimates[-1][node - 1]
    print(f"Final x{node}: {final_estimate:.2f}")
    L1_norm += abs(final_estimate-3)
    print("L1_normATP", L1_norm)

print(L1_norm)

