# -*- coding: utf-8 -*-
"""Japanese_EuSN_Mnist.ipynb
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from keras.datasets import mnist
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Paths
train_imgs_path = '/content/k49-train-imgs.npz'
train_labels_path = '/content/k49-train-labels.npz'
test_imgs_path = '/content/k49-test-imgs.npz'
test_labels_path = '/content/k49-test-labels.npz'

# Data loading
x_train = np.load(train_imgs_path)['arr_0']
y_train= np.load(train_labels_path)['arr_0']
x_test= np.load(test_imgs_path)['arr_0']
y_test= np.load(test_labels_path)['arr_0']

x_train.shape, y_train.shape

x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).long()  # labels are typically long
x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test).long()  # labels are typically long

x_train.shape, y_train.shape

# Create TensorDatasets and DataLoaders
batch_size = 128
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

class EulerReservoirCell(nn.Module):
    def __init__(self, units, activation=nn.Tanh(), weight_decay=0.01):
        super(EulerReservoirCell, self).__init__()
        self.units = units
        self.activation = activation
        self.input_kernel = nn.Linear(784 , units)
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
        self.fc = nn.Linear(units, 49)  # number of output classes

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
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)
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