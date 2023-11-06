import numpy as np

class CNN:

  def __init__(self, input_shape, num_filters):
    self.input_shape = input_shape
    self.num_filters = num_filters

    self.weights = self._initialize_weights()

  def _initialize_weights(self):

    weights = {}

    #layer 1
    weights['W1'] = np.random.randn(self.num_filters, self.input_shape[0], 3, 3) * 0.01
    weights['b1'] = np.zeros(self.num_filters)

    #layer 2
    weights['W2'] = np.random.randn(self.num_filters*2, self.num_filters, 3, 3) * 0.01 
    weights['b2'] = np.zeros(self.num_filters*2)

    #layer 3
    weights['W3'] = np.random.randn(self.num_filters*2, self.num_filters*2, 3, 3) * 0.01
    weights['b3'] = np.zeros(self.num_filters*2)

    #FC layer 
    weights['Wf'] = np.random.randn(self.num_filters * 2 * (self.input_shape[1]//8) * (self.input_shape[2]//8), 32) * 0.01
    weights['bf'] = np.zeros(32)

    return weights

  def extract_features(self, X):

    # Implement layers
    conv1 = self.conv_layer(X, self.weights['W1'], self.weights['b1']) 
    act1 = self.relu(conv1)

    conv2 = self.conv_layer(act1, self.weights['W2'], self.weights['b2'])
    act2 = self.relu(conv2)

    conv3 = self.conv_layer(act2, self.weights['W3'], self.weights['b3'])
    act3 = self.relu(conv3)

    pooled = self.maxpool_layer(act3)
    flattened = self.flatten(pooled)

    fc = flattened @ self.weights['Wf'] + self.weights['bf']
    features = self.relu(fc)
    return features

  # Helper functions
  def conv_layer(self, X, W, b):
    # Convolution operation
    conv_out = []
    for f in range(W.shape[0]):
      conv = self.conv(X, W[f], b[f])
      conv_out.append(conv)
    return np.array(conv_out)

  def conv(self, X, W, b):
    # Convolution helper 
    conv = np.zeros((X.shape[1]-W.shape[1]+1, X.shape[2]-W.shape[2]+1))
    for i in range(conv.shape[0]):
      for j in range(conv.shape[1]):
        conv[i,j] = np.sum(X[:,i:i+W.shape[1],j:j+W.shape[2]] * W) + b
    return conv

  def relu(self, X):
    return np.maximum(0, X)

  def maxpool_layer(self, X):
    return np.array([self.maxpool(x) for x in X])

  def maxpool(self, X):
    pooled = np.zeros((X.shape[0]//2, X.shape[1]//2))
    for i in range(pooled.shape[0]):
      for j in range(pooled.shape[1]):
        pooled[i,j] = np.max(X[i*2:(i+1)*2, j*2:(j+1)*2])
    return pooled

  def flatten(self, X):
    return np.array([x.flatten() for x in X])