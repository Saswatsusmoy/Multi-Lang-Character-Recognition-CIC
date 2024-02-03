"""MNIST_CNN_ESN.py
"""

import numpy as np
import pandas as pd
import seaborn as sns
import os

from sklearn.preprocessing import OneHotEncoder

from tensorflow import keras
from keras import layers
from tensorflow.keras import Sequential
from keras import Model
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization
from scipy.special import softmax

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, confusion_matrix


training_dir='mnist_train.csv'
test_dir='mnist_test.csv'

df = pd.read_csv(training_dir)

# Reading images and labels from csv
x_train= df.drop('label',axis=1)
y_train = df['label']

# Creating numpy arrays and reshaping images into 28x28 pixels
x_train = x_train.to_numpy()
y_train = y_train.to_numpy()
x_train = x_train.reshape(60000,28,28,1)

# Reading test data
df = pd.read_csv(test_dir)
x_test= df.drop('label',axis=1)
y_test = df['label']

# Creating numpy arrays and reshaping images into 28x28 pixels
x_test=x_test.to_numpy()
y_test=y_test.to_numpy()
x_test=x_test.reshape(10000,28,28,1)

# Scaling down image pixel values
x_test=x_test/255

# Scaling down image pixel values
x_train=x_train/255

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10)

# Create CNN model
model=Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(32,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

y_train.shape

history = model.fit(x_train, y_train, epochs=5,validation_data=(x_val, y_val), batch_size=128)

# Evaluate the CNN model on the test set
y_pred = model.predict(x_test)
y_pred_labels_cnn = np.argmax(y_pred, axis=1)
y_test_labels = y_test

# Classification report for CNN model on test data
accuracy_cnn_best = accuracy_score(y_test_labels, y_pred_labels_cnn)
print(f"Accuracy on the test set using CNN : {accuracy_cnn_best * 100:.2f}%")
print("Classification Report using CNN:")
print(classification_report(y_test_labels, y_pred_labels_cnn))

accuracy = accuracy_score(y_test_labels, y_pred_labels_cnn)
precision = precision_score(y_test_labels, y_pred_labels_cnn, average='macro')
recall = recall_score(y_test_labels, y_pred_labels_cnn, average='macro')
f1 = f1_score(y_test_labels, y_pred_labels_cnn, average='macro')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

plt.figure(figsize=(8, 5))
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

cm_esn_best = confusion_matrix(y_test_labels, y_pred_labels_cnn)
plt.figure(figsize=(8,8))
sns.heatmap(cm_esn_best, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(10), yticklabels=np.arange(10))
plt.title('Confusion Matrix Resnet')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

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

# Use the trained CNN as feature extractor
cnn_feature_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)
x_train_features = cnn_feature_extractor.predict(x_train)
x_test_features = cnn_feature_extractor.predict(x_test)


class MyEchoStateNetwork():
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
        outputs = encoder.fit_transform(outputs.reshape(-1, 1))

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

num_inputs = x_train_features.shape[1]

esn = MyEchoStateNetwork(num_inputs=num_inputs, num_outputs=10, reservoir_size=200, spectral_radius=0.95, sparsity=0.01, noise=0.001, teacher_forcing=True, output_activation=identity_func, inverse_output_activation=identity_func, random_state=42, verbose=False)

esn.fit(x_train_features, y_train)

# Predict with the ESN
y_pred_esn = esn.predict(x_test_features)
#y_pred_labels_esn = np.argmax(y_pred_esn, axis=0)
y_pred_labels_esn = np.reshape(y_pred_esn, (-1, 1))
y_pred_labels_esn.shape
y_pred_esn.shape

accuracy_esn = accuracy_score(y_test_labels, np.array(y_pred_labels_esn).reshape(y_test_labels.shape))

print(accuracy_esn)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
meters= []
def train_and_evaluate_esn(reservoir_size, spectral_radius, sparsity, noise):
    esn = MyEchoStateNetwork(num_inputs=num_inputs, num_outputs=10,
                             reservoir_size=int(reservoir_size), spectral_radius=spectral_radius,
                             sparsity=sparsity, noise=noise, teacher_forcing=True,
                             output_activation=identity_func, inverse_output_activation=identity_func,
                             random_state=None, verbose=False)

    esn.fit(x_train_features, y_train)
    y_pred_esn = esn.predict(x_test_features)
   # y_pred_labels_esn = np.argmax(y_pred_esn, axis=1)
    y_pred_labels_esn = np.reshape(y_pred_esn, (-1, 1))
    accuracy_esn = accuracy_score(y_test_labels, y_pred_labels_esn)
    precision = precision_score(y_test_labels, y_pred_labels_esn, average='weighted')
    recall = recall_score(y_test_labels, y_pred_labels_esn, average='weighted')
    f1 = f1_score(y_test_labels, y_pred_labels_esn, average='weighted')
    meters.append([accuracy,precision,recall,f1])
    # Print or store the metrics
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    return accuracy_esn

# Bayesian Optimization for hyperparameter tuning
pbounds = {'reservoir_size': (100, 800),
           'spectral_radius': (0, 1.5),
           'sparsity': (0.0, 1.0),
           'noise': (0.0001, 0.1)}

optimizer = BayesianOptimization(f=train_and_evaluate_esn, pbounds=pbounds, random_state=42)
optimizer.maximize(init_points=10, n_iter=50)