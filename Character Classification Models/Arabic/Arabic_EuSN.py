# -*- coding: utf-8 -*-
"""Arabic_EuSN.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Reading image data from csv
df= pd.read_csv('csvTrainImages 13440x1024.csv')
x_train = df.to_numpy().reshape(13439,32,32)
df = pd.read_csv('csvTestImages 3360x1024.csv')
x_test = df.to_numpy().reshape(3359,32,32)

# Reading image labels from csv
df= pd.read_csv('csvTrainLabel 13440x1.csv')
y_train = df.to_numpy()
df= pd.read_csv('csvTestLabel 3360x1.csv')
y_test = df.to_numpy()

x_train.shape, y_train.shape

# Normalize the data and add channel dimension
x_train = x_train.reshape((-1, 1, 32, 32)) / 255.
x_test = x_test.reshape((-1, 1, 32, 32)) / 255.

y_train = np.argmax(y_train, axis = 1)
y_test = np.argmax(y_test, axis = 1)

x_train.shape, y_train.shape

x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).long()  # labels are typically long
x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test).long()  # labels are typically long

x_train.shape, y_train.shape

# Create TensorDatasets and DataLoaders
batch_size = 32
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

class EulerReservoirCell(nn.Module):
    def __init__(self, units, activation=nn.Tanh(), weight_decay=0.01):
        super(EulerReservoirCell, self).__init__()
        self.units = units
        self.activation = activation
        self.input_kernel = nn.Linear(1024, units)
        self.kernel = nn.Linear(units, units, bias=False)
        self.recurrent_kernel = nn.Linear(units, units, bias=False)
        self.weight_decay = weight_decay

        # Initialize the recurrent kernel to be antisymmetric
        nn.init.kaiming_normal_(self.recurrent_kernel.weight)
        self.recurrent_kernel.weight.data = (self.recurrent_kernel.weight.data - self.recurrent_kernel.weight.data.t()) * 0.5

    def forward(self, inputs, states):
        h = 0.01  # Step size
        inputs = self.activation(self.input_kernel(inputs))
        net = self.kernel(inputs) + self.recurrent_kernel(states)
        output = states + h * self.activation(net)  # Correct Euler discretization
        return output, output

class ReadoutLayer(nn.Module):
    def __init__(self, units):
        super(ReadoutLayer, self).__init__()
        self.fc = nn.Linear(units, 28)  # number of output classes

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc(x)
        return x

class EuSNModel(nn.Module):
    def __init__(self, reservoir_units):
        super(EuSNModel, self).__init__()
        self.reservoir = EulerReservoirCell(reservoir_units)
        self.readout = ReadoutLayer(reservoir_units)

    def forward(self, inputs, states=None):
        inputs = inputs.view(inputs.size(0), -1)
        if states is None:
            states = torch.zeros(inputs.size(0), self.reservoir.units, device=inputs.device)
        reservoir_output, reservoir_state = self.reservoir(inputs, states)
        reservoir_output, reservoir_state = self.reservoir(inputs, states)
        logits = self.readout(reservoir_output)
        return logits

# Define model, optimizer, and loss function
model = EuSNModel(128)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Training loop using mini-batches
epochs = 10
for epoch in range(epochs):
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

correct = 0
total = 0
y_true = []
y_pred = []
with torch.no_grad():
        for i in range(len(x_test)):
            inputs = x_test[i].unsqueeze(0)  # Add batch dimension
            labels = y_test[i].unsqueeze(0)  # Add batch dimension
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
print(f'Epoch {epoch+1}/{epochs}: Loss: {loss.item()}, Accuracy: {correct / total}')

# Print final accuracy
print(f"Final Test Accuracy: {correct / total}")

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')
confusion = confusion_matrix(y_true, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'Confusion Matrix:\n {confusion}')