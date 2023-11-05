import numpy as np
from scipy.sparse import random
from sklearn.linear_model import Ridge, Lasso

class ESN:

  def __init__(self, n_inputs, n_outputs, n_reservoir=100, spectral_radius=0.95, 
               activation='tanh', input_scaling=1.0, bias_scaling=0.0,  
               ridge_param=0.001, leakage=1.0, regression='Ridge'):

    # Reservoir weights
    self.W = random(n_reservoir, n_reservoir, density=0.1, random_state=42)
    self.W *= spectral_radius / np.max(np.abs(np.linalg.eig(self.W)[0]))

    # Input and bias weights
    self.Win = input_scaling * np.random.rand(n_reservoir, n_inputs)
    self.b = bias_scaling * np.random.rand(n_reservoir)

    # Readout weights and regularization
    self.ridge_param = ridge_param
    if regression == 'Ridge':
      self.regression = Ridge(alpha=ridge_param)
    elif regression == 'Lasso':
      self.regression = Lasso(alpha=ridge_param)

    # Activation functions
    if activation == 'tanh':
      self.activation = np.tanh
    elif activation == 'relu':
      self.activation = lambda x:np.maximum(0, x)

    # Leakage rate     
    self.leakage = leakage

  def fit(self, X, y):

    states = self._generate_states(X)
    self.Wout = self.regression.fit(states, y).coef_

  def _generate_states(self, X):

    states = []
    state = np.zeros(self.W.shape[0])

    for x in X:
      state = (1 - self.leakage) * state + self.leakage * self.activation(
        np.dot(self.W, state) + np.dot(self.Win, x) + self.b)
      states.append(state)

    return np.array(states)

  def predict(self, X):

    state = np.zeros(self.W.shape[0])
    y_pred = []

    for x in X:
      state = (1 - self.leakage) * state + self.leakage * self.activation(
        np.dot(self.W, state) + np.dot(self.Win, x) + self.b)
      y_pred.append(np.dot(self.Wout, state))

    return np.array(y_pred)