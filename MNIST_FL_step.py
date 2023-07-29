#Federated Learning with MNIST
#When an edge is disconnected, agradient step is performed on itself
#Simulates lack of communication link

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split

# Set a random seed for reproducibility
torch.manual_seed(42)

# Step 1: Download the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_dataset = MNIST(root="./data", train=True, transform=transform, download=True)
mnist_train, mnist_val = random_split(mnist_dataset, [50000, 10000])  # Split into train and validation sets

disconnect_prob = 0
num_epochs = 100

# Step 2: Define the neural network architecture
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

# Step 3: Set up the clients and the server
num_clients = 5
clients = [SimpleNet() for _ in range(num_clients)]
server = SimpleNet()

# Step 4: Federated Learning with FedAvg
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

def server_aggregate(models, num_participating_clients):
    global_dict = server.state_dict()
    for param_name in global_dict:
        param_sum = sum(model[param_name] for model in models)
        global_dict[param_name] = param_sum / num_participating_clients
    server.load_state_dict(global_dict)

def run_federated_learning(disconnect_prob, num_epochs=num_epochs):
    client_data_loaders = [DataLoader(mnist_train, batch_size=32, shuffle=True) for _ in range(num_clients)]
    straggler_counts = {i + 1: 0 for i in range(num_clients)}  # Dictionary to track how many times each client becomes a straggler
    consecutive_straggler_counts = {i + 1: 0 for i in range(num_clients)}  # Dictionary to track consecutive straggler counts

    for epoch in range(num_epochs):
        # Simulate disconnected clients for this epoch
        num_participating_clients = sum(torch.rand(1) >= disconnect_prob for _ in range(num_clients))

        # Client updates
        models = []
        for i, client in enumerate(clients):
            if torch.rand(1) >= disconnect_prob:
                # Perform a gradient step if consecutive straggler count is greater than 0
                for _ in range(consecutive_straggler_counts[i + 1]):
                    model, loss = client_update(client, client_data_loaders[i])
                consecutive_straggler_counts[i + 1] = 0  # Reset consecutive straggler count
                model, loss = client_update(client, client_data_loaders[i])
                models.append(model)
            else:
                straggler_counts[i + 1] += 1
                consecutive_straggler_counts[i + 1] += 1

        # Server aggregates models
        if models:
            server_aggregate(models, num_participating_clients)

    # Calculate individual accuracies and final accuracy of the federated learning network
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

    # Calculate final accuracy of the federated learning network as the average of individual accuracies
    final_accuracy = sum(client_accuracies) / len(client_accuracies)

    return client_accuracies, final_accuracy, straggler_counts

# Run the federated learning simulation
client_accuracies, final_accuracy, straggler_counts = run_federated_learning(disconnect_prob)

# Print results
print("Individual Accuracies of Clients:")
for i, acc in enumerate(client_accuracies, 1):
    print(f"Client {i}: {acc:.2f}%")

print("\nFinal Accuracy of the Federated Learning Network:")
print(f"{final_accuracy:.2f}%")

# Print the count of times each client became a straggler as a dictionary
print("\nStraggler Counts:")
print(straggler_counts)
