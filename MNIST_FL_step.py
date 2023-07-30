#Simulates lack of computation power
#Federated Learning with MNIST
#When an edge is disconnected, a gradient step is still performed on itself

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import random

seed = 42
torch.manual_seed(seed)
random.seed(seed)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_dataset = MNIST(root="./data", train=True, transform=transform, download=True)
mnist_train, mnist_val = random_split(mnist_dataset, [50000, 10000])

disconnect_prob = 0.4
num_epochs = 100

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def create_imbalanced_subset(dataset, proportions):
    class_counts = {i: 0 for i in range(10)}
    subset_indices = []

    for index in range(len(dataset)):
        _, label = dataset[index]
        if class_counts[label] < proportions[label]:
            subset_indices.append(index)
            class_counts[label] += 1

    return torch.utils.data.Subset(dataset, subset_indices)

def create_random_imbalanced_subset(dataset, total_samples):
    num_classes = 10
    proportions = [total_samples // num_classes] * num_classes

    remaining_samples = total_samples - sum(proportions)
    for _ in range(remaining_samples):
        proportions[random.randint(0, num_classes - 1)] += 1

    return create_imbalanced_subset(dataset, proportions)

def run_federated_learning_with_random_heterogeneous_data(disconnect_prob, num_epochs=num_epochs):
    total_samples_per_client = [4000, 4000, 4000, 4000, 4000]
    num_clients = len(total_samples_per_client)
    avg_samples_per_client = sum(total_samples_per_client) // len(total_samples_per_client)
    remaining_samples = sum(total_samples_per_client) - (avg_samples_per_client * len(total_samples_per_client))

    for i in range(len(total_samples_per_client)):
        total_samples_per_client[i] += remaining_samples // len(total_samples_per_client)
        remaining_samples -= remaining_samples // len(total_samples_per_client)

    client_data_loaders = []
    num_clients = len(total_samples_per_client)
    for seed, total_samples in enumerate(total_samples_per_client, start=1):
        imbalanced_subset = create_random_imbalanced_subset(mnist_train, total_samples)
        data_loader = DataLoader(imbalanced_subset, batch_size=32, shuffle=True)
        client_data_loaders.append(data_loader)

    clients = [SimpleNet() for _ in range(num_clients)]
    server = SimpleNet()

    # Initialize server state_dict with a client's state_dict
    server.load_state_dict(clients[0].state_dict())

    # Added 'client_accuracies' to the return values
    client_accuracies, final_accuracy, straggler_counts = run_federated_learning(disconnect_prob, num_clients, clients, server, client_data_loaders, total_samples_per_client)

    # Returning 'client_accuracies' along with other results
    return client_accuracies, final_accuracy, straggler_counts, total_samples_per_client

def client_update(client, data_loader, learning_rate=0.01):
    optimizer = optim.SGD(client.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    for inputs, labels in data_loader:
        optimizer.zero_grad()
        outputs = client(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return client.state_dict(), total_loss / len(data_loader)

def server_aggregate(server, models, num_participating_clients):
    global_dict = server.state_dict()
    for param_name in global_dict:
        if param_name.endswith("weight") or param_name.endswith("bias"):
            param_sum = sum(model[param_name] for model in models)
            # Take the average instead of summing up
            global_dict[param_name] = param_sum / num_participating_clients
    server.load_state_dict(global_dict)

def run_federated_learning(disconnect_prob, num_clients, clients, server, client_data_loaders, total_samples_per_client, num_epochs=num_epochs):
    straggler_counts = {i + 1: 0 for i in range(num_clients)}
    consecutive_straggler_counts = {i + 1: 0 for i in range(num_clients)}

    for epoch in range(num_epochs):
        models = []
        participating_clients = []  # Keep track of clients that participate in federated averaging

        for i, client in enumerate(clients):
            if torch.rand(1) >= disconnect_prob:
                # Update client asynchronously
                for _ in range(consecutive_straggler_counts[i + 1]):
                    model, loss = client_update(client, client_data_loaders[i])
                consecutive_straggler_counts[i + 1] = 0
                model, loss = client_update(client, client_data_loaders[i])
                models.append(model)
                participating_clients.append(client)  # Include the client for federated averaging
            else:
                straggler_counts[i + 1] += 1
                consecutive_straggler_counts[i + 1] += 1

        # Perform federated averaging with participating clients' models
        if participating_clients:
            num_participating_clients = len(participating_clients)
            server_aggregate(server, models, num_participating_clients)
            # Update clients with the aggregated model from the server
            for client in participating_clients:
                client.load_state_dict(server.state_dict())

    client_accuracies = []
    correct_total = 0
    total_samples = len(mnist_val)

    for i, client in enumerate(clients):
        correct = 0
        with torch.no_grad():
            for inputs, labels in DataLoader(mnist_val, batch_size=32):
                outputs = client(inputs)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

        client_accuracy = 100.0 * correct / total_samples
        client_accuracies.append(client_accuracy)

    final_accuracy = sum(client_accuracies) / len(client_accuracies)

    return client_accuracies, final_accuracy, straggler_counts

# Run the federated learning simulation
client_accuracies, final_accuracy, straggler_counts, total_samples_per_client = run_federated_learning_with_random_heterogeneous_data(disconnect_prob)

# Print results
print("Individual Accuracies of Clients:")
for i, acc in enumerate(client_accuracies, 1):
    print(f"Client {i}: {acc:.2f}%")

print("\nMean Accuracy:")
print(f"{final_accuracy:.2f}%")

# Print the count of times each client became a straggler as a dictionary
print("\nStraggler Counts:")
print(straggler_counts)

# Print the total samples per client
print("\nTotal Samples Per Client:")
print(total_samples_per_client)
