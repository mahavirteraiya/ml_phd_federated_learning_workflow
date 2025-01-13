import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Define the global model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        return self.fc(x)

# Simulate data for clients
def generate_client_data(num_clients, samples_per_client):
    client_data = []
    for _ in range(num_clients):
        x = torch.randn(samples_per_client, 2)
        y = (x[:, 0] * 2 + x[:, 1] * 3 + torch.randn(samples_per_client) * 0.1).unsqueeze(1)
        client_data.append(TensorDataset(x, y))
    return client_data

# Local training on client devices
def train_local_model(model, data_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        for x_batch, y_batch in data_loader:
            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

# Federated averaging to update global model
def federated_averaging(global_model_state_dict, local_models):
    new_global_state_dict = {}
    with torch.no_grad():
        for param_name in global_model_state_dict.keys():
            param_sum = sum(local_model.state_dict()[param_name] for local_model in local_models)
            new_global_state_dict[param_name] = param_sum / len(local_models)
    return new_global_state_dict

# Differential privacy mechanism: Add noise to model updates
def add_differential_privacy_noise(model_state_dict, epsilon=0.5):
    noisy_state_dict = {}
    for param_name, param_tensor in model_state_dict.items():
        noise = torch.normal(0, 1/epsilon, size=param_tensor.size())
        noisy_state_dict[param_name] = param_tensor + noise
    return noisy_state_dict

# Main federated learning workflow
def federated_learning_workflow(num_clients=10, samples_per_client=100, rounds=5):
    # Initialize global model and client models
    global_model = SimpleModel()
    client_data = generate_client_data(num_clients, samples_per_client)
    local_models = [SimpleModel() for _ in range(num_clients)]

    # Define loss function and optimizer
    criterion = nn.MSELoss()

    for round_num in range(rounds):
        print(f"--- Round {round_num + 1} ---")
        
        # Train local models
        for i, local_model in enumerate(local_models):
            local_model.load_state_dict(global_model.state_dict())  # Start with global weights
            optimizer = optim.SGD(local_model.parameters(), lr=0.01)
            data_loader = DataLoader(client_data[i], batch_size=32, shuffle=True)
            train_local_model(local_model, data_loader, criterion, optimizer)

        # Aggregate local models into global model
        global_model_state_dict = federated_averaging(global_model.state_dict(), local_models)
        
        # Apply differential privacy to the global model updates
        global_model_state_dict = add_differential_privacy_noise(global_model_state_dict)

        # Update the global model with aggregated parameters
        global_model.load_state_dict(global_model_state_dict)

        # Evaluate global model (optional step: use validation/test dataset here if available)
        print(f"Round {round_num + 1} completed.")

# Run the federated learning workflow
if __name__ == "__main__":
    federated_learning_workflow()
