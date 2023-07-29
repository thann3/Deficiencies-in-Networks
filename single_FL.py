#Federated Learning with single-variable loss function
#When an edge is disconnected, no gradient step is not performed on itself
#Simulates lack of computation power

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, datasets

# Set random seed for reproducibility
np.random.seed(42)

# Define the negative log likelihood cost function
def cost_function(x, c):
    return tf.keras.losses.sparse_categorical_crossentropy(c, x)

# Define the standard FedAvg algorithm
def federated_averaging(models):
    if len(models) > 0:
        averaged_weights = []
        for layer_weights in zip(*[model.get_weights() for model in models]):
            averaged_layer_weights = tf.reduce_mean(layer_weights, axis=0)
            averaged_weights.append(averaged_layer_weights.numpy())
        return averaged_weights
    else:
        return None

# Function to simulate a round of asynchronous federated learning
def train_one_round(global_model, nodes, data_by_node, learning_rate, probability_to_disconnect, straggler_counts):
    # Perform updates for each node
    for node_idx, node in enumerate(nodes):
        if np.random.rand() > probability_to_disconnect:
            x_train, y_train = data_by_node[node_idx]
            with tf.GradientTape() as tape:
                logits = node(x_train)
                loss = tf.reduce_mean(cost_function(logits, y_train))
            gradients = tape.gradient(loss, node.trainable_variables)
            node.optimizer.apply_gradients(zip(gradients, node.trainable_variables))
        else:
            # Node is a straggler, increase the count in the dictionary
            straggler_counts[node_idx] += 1

    # Aggregate models using FedAvg
    models = [node for node_idx, node in enumerate(nodes) if np.random.rand() > probability_to_disconnect]
    aggregated_model_weights = federated_averaging(models)
    if aggregated_model_weights is not None:
        global_model.set_weights(aggregated_model_weights)

    # Send the updated global model back to all nodes
    for node in nodes:
        if np.random.rand() > probability_to_disconnect:
            node.set_weights(global_model.get_weights())

# Function to run multiple rounds of asynchronous federated learning
def run_federated_learning(num_rounds, probability_to_disconnect, learning_rate):
    # Download and preprocess the MNIST dataset
    (x_train, y_train), _ = datasets.mnist.load_data()
    x_train = x_train / 255.0  # Normalize pixel values to [0, 1]
    x_train = x_train[..., np.newaxis]  # Add a channel dimension for CNN input
    num_nodes = 5  # Number of nodes participating in federated learning
    data_per_node = len(x_train) // num_nodes
    data_by_node = [(x_train[i * data_per_node: (i + 1) * data_per_node],
                     y_train[i * data_per_node: (i + 1) * data_per_node])
                    for i in range(num_nodes)]

    # Create and compile the global model
    global_model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    global_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Create and compile individual nodes with separate data
    nodes = [models.clone_model(global_model) for _ in range(num_nodes)]
    for node in nodes:
        node.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Initialize the dictionary to keep track of straggler counts for each node
    straggler_counts = {i: 0 for i in range(num_nodes)}

    # Print the initial accuracy of the nodes
    initial_accs = [node.evaluate(data_by_node[i][0], data_by_node[i][1], verbose=0)[1] for i, node in enumerate(nodes)]
    print("Initial accuracies:", initial_accs)

    for round in range(num_rounds):
        train_one_round(global_model, nodes, data_by_node, learning_rate, probability_to_disconnect, straggler_counts)

    # Get the final accuracy of the nodes and the accuracy of the global model
    final_accs = [node.evaluate(data_by_node[i][0], data_by_node[i][1], verbose=0)[1] for i, node in enumerate(nodes)]
    global_acc = global_model.evaluate(x_train, y_train, verbose=0)[1]

    print("Final accuracies:", final_accs)
    print("Global model accuracy:", global_acc)
    print("Average accuracy of nodes:", np.mean(final_accs))

    # Print the straggler counts for each node
    print("Straggler counts:", straggler_counts)

run_federated_learning(num_rounds=100, probability_to_disconnect=0.1, learning_rate=0.01)
