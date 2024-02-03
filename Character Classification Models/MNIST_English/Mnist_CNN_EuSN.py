# -*- coding: utf-8 -*-
"""MNIST_EuSN+CNN.ipynb
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

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train.shape, y_train.shape

# Convert the dataset to NumPy array and normalize the data
X_train = x_train.reshape((-1, 1, 28, 28)).astype(np.float32) / 255.
X_test = x_test.reshape((-1, 1, 28, 28)).astype(np.float32) / 255.
y_train = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)

# Convert the datasets to PyTorch tensors
x_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)
x_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test)

# Create TensorDatasets and DataLoaders
batch_size = 128
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # Third convolutional layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(6272, 128)

    def forward(self, x):
        # Apply first convolution, activation function, and pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Apply second convolution and activation function
        x = F.relu(self.conv2(x))
        # Apply third convolution, activation function, and pooling
        x = self.pool(F.relu(self.conv3(x)))
        return x

class EulerReservoirCell(nn.Module):
    # Assuming the output of FeatureExtractor is (batch_size, 32, 14, 14)
    def __init__(self, units, input_size=32*14*14, activation=nn.Tanh(), weight_decay=0.01):
        super(EulerReservoirCell, self).__init__()
        self.units = units
        self.activation = activation
        self.input_kernel = nn.Linear(input_size, units)
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
        self.fc = nn.Linear(units, 10)  # Assuming 10 is the number of output classes

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc(x)
        return x

class EuSNModel(nn.Module):
    def __init__(self, reservoir_units, feature_extractor):
        super(EuSNModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.reservoir = EulerReservoirCell(reservoir_units)
        self.readout = ReadoutLayer(reservoir_units)

    def forward(self, inputs, states=None):
        features = self.feature_extractor(inputs)
        features = features.view(features.size(0), -1)  # Flatten the features
        if states is None:
            states = torch.zeros(features.size(0), self.reservoir.units, device=features.device)
        reservoir_output, reservoir_state = self.reservoir(features, states)
        logits = self.readout(reservoir_output)
        return logits

# Define the feature extractor, model, optimizer, and loss function
feature_extractor = FeatureExtractor()
model = EuSNModel(128, feature_extractor)
model = model.to(device)
optimizer = optim.Adam(list(feature_extractor.parameters()) + list(model.parameters()), lr=0.001)
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
with torch.no_grad():
    for i in range(len(x_test)):
        inputs = x_test[i].unsqueeze(0)
        labels = y_test[i].unsqueeze(0)
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Epoch {epoch+1}/{epochs}: Loss: {loss.item()}, Accuracy: {correct / total}')

y_true = []
y_pred = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

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

