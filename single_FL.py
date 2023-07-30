#Federated Learning with single-variable loss function
#When an edge is disconnected, no gradient step is not performed on itself
#Simulates lack of computation power

import numpy as np
import tensorflow as tf

#Ensure reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

num_rounds = 10000
probability_to_disconnect = 0
learning_rate = 0.01

# Define the cost function: (x_i - c_i)^2
def cost_function(x, c):
    return tf.square(x - c)

# Define the standard FedAvg algorithm
def federated_averaging(models):
    if len(models) > 0:
        return np.mean(models, axis=0)
    else:
        return None

# Initialize nodes and server
c_values = np.array([1, 2, 3, 4, 5], dtype=np.float32)

num_nodes = 5
nodes = [tf.Variable(np.random.randn(1).astype(np.float32)) for _ in range(num_nodes)]
global_model = tf.Variable(np.random.randn(1).astype(np.float32))

# Function to simulate a round of asynchronous federated learning
def train_one_round(global_model, nodes, c_values, learning_rate, probability_to_disconnect, straggler_counts):
    # Perform updates for each node
    disconnected_nodes = []  # Store disconnected nodes
    for node in nodes:
        if np.random.rand() > probability_to_disconnect:
            with tf.GradientTape() as tape:
                cost = cost_function(node, c_values[nodes.index(node)])
            gradients = tape.gradient(cost, node)
            node.assign_sub(learning_rate * gradients)
        else:
            # Node is disconnected, add to the disconnected_nodes list
            disconnected_nodes.append(node)
            # Increment the straggler count for the disconnected node
            straggler_counts[nodes.index(node)] += 1  # Update the straggler count

    # Aggregate models using FedAvg
    models = [node.numpy() for node in nodes if np.random.rand() > probability_to_disconnect]
    aggregated_model = federated_averaging(models)
    if aggregated_model is not None:
        global_model.assign(aggregated_model)

    # Send the updated global model back to all nodes
    for node in nodes:
        if np.random.rand() > probability_to_disconnect:
            node.assign(global_model)

    # Perform gradient steps on disconnected nodes using the global model after aggregation
    for disconnected_node in disconnected_nodes:
        if np.random.rand() > probability_to_disconnect:
            with tf.GradientTape() as tape:
                cost = cost_function(disconnected_node, c_values[nodes.index(disconnected_node)])
            gradients = tape.gradient(cost, disconnected_node)
            disconnected_node.assign_sub(learning_rate * gradients)

# Function to run multiple rounds of asynchronous federated learning
def run_federated_learning(num_rounds, probability_to_disconnect, learning_rate):
    # Initialize the dictionary to keep track of straggler counts for each node
    straggler_counts = {i: 0 for i in range(num_nodes)}

    # Print the initial estimates of the nodes
    initial_x_values = [node.numpy()[0] for node in nodes]
    print("Initial x values:", initial_x_values)

    for round in range(num_rounds):
        train_one_round(global_model, nodes, c_values, learning_rate, probability_to_disconnect, straggler_counts)

    # Get the final x values and the optimal solution x*
    final_x_values = [node.numpy()[0] for node in nodes]
    optimal_x = 3

    print("Final x values:", final_x_values)

    mean_x = 0
    for i in final_x_values:
        mean_x += i
    mean_x /= len(final_x_values)

    print("Mean x value:", mean_x)

    print("Optimal solution:", optimal_x)

    # Calculate the sum of the L1 norms of the final values with respect to x=3
    l1_norm_sum = np.sum([abs(final_x - optimal_x) for final_x in final_x_values])

    # Print the sum of the L1 norms of the final values
    print("L1 norm:", l1_norm_sum)

    # Print the straggler counts for each node after the rounds are completed
    print("Straggler counts:", straggler_counts)

run_federated_learning(num_rounds, probability_to_disconnect, learning_rate)
