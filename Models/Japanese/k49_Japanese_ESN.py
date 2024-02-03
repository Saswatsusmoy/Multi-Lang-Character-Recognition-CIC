"""K49_JAPANESE_ESN_final.ipynb
"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from bayes_opt import BayesianOptimization

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

x_train = x_train.reshape(232365,784)
x_test = x_test.reshape(38547,784)

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

pca = PCA(n_components=50)  # Adjust n_components as needed
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

meters= []
def optimize_esn(reservoir_size, spectral_radius, sparsity, noise):
    # Train the model with the given hyperparameters
    esn = MyEchoStateNetworkWrapper(num_inputs=50, num_outputs=49, reservoir_size=int(reservoir_size), spectral_radius=spectral_radius, sparsity=sparsity, noise=noise)  # Adjust num_inputs to match the number of PCA components

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
hyperparameters_bounds = {'reservoir_size': (100, 200), 'spectral_radius': (0, 1.5), 'sparsity': (0, 0.5), 'noise': (0, 0.1)}

# Initialize the optimizer
optimizer = BayesianOptimization(f=optimize_esn, pbounds=hyperparameters_bounds, random_state=42)

# Optimize
optimizer.maximize(init_points=20, n_iter=50)

# Accuracy: 0.458738682647158,
#  Precision: 0.5023550030390262,
#   Recall: 0.458738682647158,
#   F1 Score: 0.435727474525001