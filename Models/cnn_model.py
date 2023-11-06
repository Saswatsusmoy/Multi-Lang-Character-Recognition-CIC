import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_recall_fscore_support  # noqa: E501
import matplotlib.pyplot as plt

from cnn import CNN


# Load dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X_train = train.drop('label', axis=1).values.reshape(-1, 28, 28)
y_train = train['label'].values

X_test = test.drop('label', axis=1).values.reshape(-1, 28, 28) 
y_test = test['label'].values

# Scale pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Initialize CNN  
input_shape = (28, 28)
num_filters = 32
cnn = CNN(input_shape, num_filters)

# Train model
cnn.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate model
y_pred = cnn.predict(X_test)

print('Accuracy:', accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred, 
                            target_names=['Class 0', 'Class 1', 'Class 2']))

# Calculate F1 score per class
f1 = f1_score(y_test, y_pred, average=None) 

# Calculate precision, recall and F1 weighted averages
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

print('F1 scores: ', f1)
print('Precision: {:.2f}'.format(precision))
print('Recall: {:.2f}'.format(recall))
print('F1 score: {:.2f}'.format(f1))

# Confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
# Classification report
print(classification_report(y_test, y_pred)) 

# Plot confusion matrix
plt.matshow(conf_mat)
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('conf_mat.png')