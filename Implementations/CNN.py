import numpy as np

# CNN model
class CNN:
  def __init__(self, input_shape, num_filters, filter_size, pool_size, num_classes):
    # CNN architecture details
    
    # Initialize weights
    self.W1 = np.random.rand(num_filters, input_shape[0], filter_size, filter_size)
    self.b1 = np.zeros(num_filters)
    
    self.W2 = np.random.rand(num_filters * (input_shape[1]//pool_size) * (input_shape[2]//pool_size), num_classes)   # noqa: E501
    self.b2 = np.zeros(num_classes)

  def extract_features(self, X):
    # Implement CNN layers
    conv_out = self.conv_layer(X)
    pooled_out = self.maxpool_layer(conv_out)
    flattened = self.flatten(pooled_out)
    return flattened
  
  # Helper functions
  def conv_layer(self, X):
    # Convolution operation
    conv_out = []
    for f in range(self.num_filters):
      conv = np.zeros((X.shape[1]-self.filter_size+1, X.shape[2]-self.filter_size+1))
      for i in range(conv.shape[0]):
        for j in range(conv.shape[1]):
          conv[i,j] = np.sum(X[:,i:i+self.filter_size,j:j+self.filter_size] * self.W1[f]) + self.b1[f]  # noqa: E501
      conv_out.append(conv)
    return np.array(conv_out)

  def maxpool_layer(self, X):
    # Max pooling operation
    return np.array([self.maxpool(x) for x in X])

  def maxpool(self, X):
    # Max pooling helper
    pooled = np.zeros((X.shape[0]//self.pool_size, X.shape[1]//self.pool_size))
    for i in range(pooled.shape[0]):
      for j in range(pooled.shape[1]):
        pooled[i,j] = np.max(X[i*self.pool_size:(i+1)*self.pool_size, 
                              j*self.pool_size:(j+1)*self.pool_size])
    return pooled

  def flatten(self, X):
    # Flatten output for fully connected layer
    return np.array([x.flatten() for x in X])