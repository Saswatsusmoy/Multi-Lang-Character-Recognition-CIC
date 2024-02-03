# -*- coding: utf-8 -*-
"""Arabic_ESN.ipynb
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.special import softmax
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from bayes_opt import BayesianOptimization

import pandas as pd
import tensorflow
from tensorflow import keras
from keras import Model
from keras import layers
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA

# Reading image data from csv

df= pd.read_csv('/content/csvTrainImages 13440x1024.csv')
x_train = df.to_numpy().reshape(13439,1024)
df = pd.read_csv('/content/csvTestImages 3360x1024.csv')
x_test = df.to_numpy().reshape(3359,1024)

# Reading image labels from csv

df= pd.read_csv('/content/csvTrainLabel 13440x1.csv')
y_train = df.to_numpy()
df= pd.read_csv('/content/csvTestLabel 3360x1.csv')
y_test = df.to_numpy()

# Scale down image pixel values

x_train = x_train/255
x_test = x_test/255

# Adjust labels according to CNN model

y_train=y_train -1
y_test = y_test -1
# y_train = np.argmax(y_train, axis=1)
# y_test = np.argmax(y_test,axis=1)

print(y_train.shape)
print(x_train.shape)

# Splitting training data into training data and validation data

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10)

def adjust_dimensions(val, target_length):
    if val is not None:
        val = np.array(val)
        if val.ndim == 0:
            val = np.array([val] * target_length)
        elif val.ndim == 1:
            if not len(val) == target_length:
                raise ValueError("Argument must have length " + str(target_length))
        else:
            raise ValueError("Invalid argument")
    return val

def identity_func(x):
    return x

class MyEchoStateNetworkWrapper():
    def __init__(self, num_inputs, num_outputs, reservoir_size=200,
                 spectral_radius=0.95, sparsity=0, noise=0.001, input_shift=None,
                 input_scaling=None, teacher_forcing=True, feedback_scaling=None,
                 teacher_scaling=None, teacher_shift=None,
                 output_activation=identity_func, inverse_output_activation=identity_func,
                 random_state=None, verbose=False):
        self.num_inputs = num_inputs
        self.reservoir_size = reservoir_size
        self.num_outputs = num_outputs
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise
        self.input_shift = adjust_dimensions(input_shift, num_inputs)
        self.input_scaling = adjust_dimensions(input_scaling, num_inputs)
        self.teacher_scaling = teacher_scaling
        self.teacher_shift = teacher_shift
        self.feedback_scaling = feedback_scaling  # Add this line
        self.output_activation = output_activation
        self.inverse_output_activation = inverse_output_activation
        self.random_state = random_state

        if isinstance(random_state, np.random.RandomState):
            self.random_state_ = random_state
        elif random_state:
            try:
                self.random_state_ = np.random.RandomState(random_state)
            except TypeError as e:
                raise Exception("Invalid seed: " + str(e))
        else:
            self.random_state_ = np.random.mtrand._rand

        self.teacher_forcing = teacher_forcing
        self.verbose = verbose
        self.initialize_weights()

    def initialize_weights(self):
        W = self.random_state_.rand(self.reservoir_size, self.reservoir_size) - 0.5
        W[self.random_state_.rand(*W.shape) < self.sparsity] = 0
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = W * (self.spectral_radius / radius)
        self.W_in = self.random_state_.rand(
            self.reservoir_size, self.num_inputs) * 2 - 1
        self.W_feedback = self.random_state_.rand(
            self.reservoir_size, self.num_outputs) * 2 - 1

    def update_state(self, state, input_pattern, output_pattern):
        if self.teacher_forcing:
            preactivation = (np.dot(self.W, state)
                             + np.dot(self.W_in, input_pattern)
                             + np.dot(self.W_feedback, output_pattern))
        else:
            preactivation = (np.dot(self.W, state)
                             + np.dot(self.W_in, input_pattern))
        return (np.tanh(preactivation)
                + self.noise * (self.random_state_.rand(self.reservoir_size) - 0.5))

    def scale_inputs(self, inputs):
        if self.input_scaling is not None:
            inputs = np.dot(inputs, np.diag(self.input_scaling))
        if self.input_shift is not None:
            inputs = inputs + self.input_shift
        return inputs

    def scale_teacher(self, teacher):
        if self.teacher_scaling is not None:
            teacher = teacher * self.teacher_scaling
        if self.teacher_shift is not None:
            teacher = teacher + self.teacher_shift
        return teacher

    def unscale_teacher(self, teacher_scaled):
        if self.teacher_shift is not None:
            teacher_scaled = teacher_scaled - self.teacher_shift
        if self.teacher_scaling is not None:
            teacher_scaled = teacher_scaled / self.teacher_scaling
        return teacher_scaled

    def fit(self, inputs, outputs):
        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (len(inputs), -1))
        if outputs.ndim < 2:
            outputs = np.reshape(outputs, (len(outputs), -1))

        # One-hot encode the outputs
        encoder = OneHotEncoder(sparse=False)
        # outputs = encoder.fit_transform(outputs.reshape(-1, 1))

        inputs_scaled = self.scale_inputs(inputs)
        teachers_scaled = self.scale_teacher(outputs)

        states = np.zeros((inputs.shape[0], self.reservoir_size))
        for n in range(1, inputs.shape[0]):
            states[n, :] = self.update_state(states[n - 1], inputs_scaled[n, :],
                                             teachers_scaled[n - 1, :])

        transient = min(int(inputs.shape[1] / 10), 100)
        extended_states = np.hstack((states, inputs_scaled))
        self.W_out = np.dot(np.linalg.pinv(extended_states[transient:, :]),
                            self.inverse_output_activation(teachers_scaled[transient:, :])).T

        return self

    def predict(self, inputs):
      if inputs.ndim < 2:
        inputs = np.reshape(inputs, (len(inputs), -1))

      n_samples = inputs.shape[0]

      last_state = np.zeros(self.reservoir_size)
      last_input = np.zeros(self.num_inputs)
      last_output = np.zeros(self.num_outputs)

      inputs = np.vstack([last_input, self.scale_inputs(inputs)])
      states = np.vstack([last_state, np.zeros((n_samples, self.reservoir_size))])
      outputs = np.vstack([last_output, np.zeros((n_samples, self.num_outputs))])

      for n in range(n_samples):
        states[n + 1, :] = self.update_state(states[n, :], inputs[n + 1, :], outputs[n, :])
        outputs[n + 1, :] = self.output_activation(np.dot(self.W_out, np.concatenate([states[n + 1, :], inputs[n + 1, :]])))

      # Apply softmax to the outputs and return the class with the highest probability
      probabilities = softmax(outputs[1:], axis=1)
      return np.argmax(probabilities, axis=1)

x_train.shape

pca = PCA(n_components=50)  # Adjust n_components as needed
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
meters= []
def optimize_esn(reservoir_size, spectral_radius, sparsity, noise):
    # Train the model with the given hyperparameters
    esn = MyEchoStateNetworkWrapper(num_inputs=50, num_outputs=28, reservoir_size=int(reservoir_size), spectral_radius=spectral_radius, sparsity=sparsity, noise=noise)  # Adjust num_inputs to match the number of PCA components

    esn.fit(x_train, y_train)

    # Make predictions on the test set
    y_pred = esn.predict(x_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    meters.append([accuracy,precision,recall,f1])

    # Print or store the metrics
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

    return accuracy  # Return the value you want to maximize/minimize

# Define the bounds of the hyperparameters to optimize
hyperparameters_bounds = {'reservoir_size': (100, 1000), 'spectral_radius': (0, 1.5), 'sparsity': (0, 1.0), 'noise': (0, 0.1)}

print(x_train.shape,y_train.shape)

# Initialize the optimizer
optimizer = BayesianOptimization(f=optimize_esn, pbounds=hyperparameters_bounds, random_state=42)

y_train=to_categorical(y_train)

y_train.shape

# Optimize
optimizer.maximize(init_points=30, n_iter=50)

print(optimizer.max)

#Accuracy: 0.4867520095266448,
#Precision: 0.48125942071499533,
#Recall: 0.4867520095266448,
#F1 Score: 0.4789166537940101