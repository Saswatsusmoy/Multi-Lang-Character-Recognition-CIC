import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support  # noqa: E501

from esn import ESN

# Load data
X_train = pd.read_csv('train.csv')  
X_test = pd.read_csv('test.csv')

# Extract labels
y_train = X_train['label']
y_test = X_test['label'] 

# Remove label column
X_train.drop('label', axis=1, inplace=True)  
X_test.drop('label', axis=1, inplace=True)

# Normalize pixel values 0-255 to 0-1
X_train = X_train / 255.0
X_test = X_test / 255.0

# Create model
esn = ESN(n_inputs=28*28, n_outputs=10)

# Reshape to vectors
X_train = X_train.to_numpy().reshape(-1, 28*28)
X_test = X_test.to_numpy().reshape(-1, 28*28)

# Train, predict and evaluate
esn.fit(X_train, y_train)
y_pred = esn.predict(X_test)


test_acc = np.mean(y_pred == y_test)
print('Accuracy:', test_acc)

precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
print('F-Score:', fscore)
print('Precision:', precision)
print('Recall:', recall)

f1 = f1_score(y_test, y_pred, average='weighted')  
print('F1 Score:', f1)

print(classification_report(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
im = ax.imshow(cm)

ax.set_xlabel('Predicted')
ax.set_ylabel('True')
fig.tight_layout()
plt.show()

# Plot reservoir activations
plt.plot(esn.activation_history.T) 
plt.xlabel('Time')
plt.ylabel('Neuron activation')
plt.show()