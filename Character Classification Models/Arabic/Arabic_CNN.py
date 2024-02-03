# -*- coding: utf-8 -*-
"""Arabic_CNN_Final.ipynb
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import Model
from keras import layers
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import seaborn as sns

# Reading image data from csv
df= pd.read_csv('/content/csvTrainImages 13440x1024.csv')
x_train = df.to_numpy().reshape(13439,32,32)
df = pd.read_csv('/content/csvTestImages 3360x1024.csv')
x_test = df.to_numpy().reshape(3359,32,32)

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

print(y_train.shape)
print(x_train.shape)

# Splitting training data into training data and validation data
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10)

# Creating CNN model
model = Sequential()

model.add(layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu', input_shape=(32,32,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.3))

model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.3))

model.add(layers.Conv2D(96, (3, 3), strides=(1, 1), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.3))

model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(28, activation='softmax'))


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Training CNN model
history=model.fit(x_train, y_train, epochs=10,validation_data=(x_val, y_val), batch_size=32)

# Evaluate the CNN model on the test set

y_pred_cnn = model.predict(x_test)
y_pred_labels_cnn = np.argmax(y_pred_cnn, axis=1)
y_test_labels = np.argmax(y_test, axis=0)
y_test_labels=y_test

# Classification report for CNN model on test data

accuracy_esn_best = accuracy_score(y_test_labels, y_pred_labels_cnn)
print(f"Accuracy on the test set using CNN : {accuracy_esn_best * 100:.2f}%")
print("Classification Report using CNN:")
print(classification_report(y_test_labels, y_pred_labels_cnn))

# Calculate metrics
accuracy = accuracy_score(y_test_labels, y_pred_labels_cnn)
precision = precision_score(y_test_labels, y_pred_labels_cnn, average='macro')
recall = recall_score(y_test_labels, y_pred_labels_cnn, average='macro')
f1 = f1_score(y_test_labels, y_pred_labels_cnn, average='macro')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Confusion Matrix for CNN
cm_esn_best = confusion_matrix(y_test_labels, y_pred_labels_cnn)
plt.figure(figsize=(12,12))
sns.heatmap(cm_esn_best, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(28), yticklabels=np.arange(28))
plt.title('Confusion Matrix CNN')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

#Plotting the loss graph
plt.figure(figsize=(8, 5))
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()