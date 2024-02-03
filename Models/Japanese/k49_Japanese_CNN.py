# -*- coding: utf-8 -*-
"""K49_JAPANESE_CNN_final.ipynb
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras import layers
from  sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from keras import Sequential
from sklearn.model_selection import train_test_split
import seaborn as sns

# Defining data directories

train_imgs_path = '/content/k49-train-imgs.npz'
train_labels_path = '/content/k49-train-labels.npz'
test_imgs_path = '/content/k49-test-imgs.npz'
test_labels_path = '/content/k49-test-labels.npz'

# Data loading
x_train = np.load(train_imgs_path)['arr_0']
y_train= np.load(train_labels_path)['arr_0']
x_test= np.load(test_imgs_path)['arr_0']
y_test= np.load(test_labels_path)['arr_0']

# Reshaping images in 28x28 pixels

x_train = x_train.reshape(232365,28,28,1)
x_test = x_test.reshape(38547,28,28,1)

# Splitting dataset into training and validation set
x_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10)

# One hot encoding images label

y_train = keras.utils.to_categorical(y_train)
y_test= keras.utils.to_categorical(y_test)
y_val=keras.utils.to_categorical(y_val)
y_test.shape

# Creating CNN model

model = Sequential()

model.add(layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu', input_shape=(28,28,1)))
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
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(49, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# printing summary of the model
model.summary()

#Training neural network model for 5 epochs using training data, validating after each epoch, with a batch size of 128.
history = model.fit(x_train, y_train, epochs=10,validation_data=(X_val, y_val), batch_size=128)

# Evaluate the CNN model on the test set
y_pred = model.predict(x_test)
y_pred_labels_cnn = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Classification report for CNN model on test data
accuracy_cnn_best = accuracy_score(y_test_labels, y_pred_labels_cnn)
print(f"Accuracy on the test set using CNN : {accuracy_cnn_best * 100:.2f}%")
print("Classification Report using CNN:")
print(classification_report(y_test_labels, y_pred_labels_cnn))

#calculating metrics
accuracy = accuracy_score(y_test_labels, y_pred_labels_cnn)
precision = precision_score(y_test_labels, y_pred_labels_cnn, average='macro')
recall = recall_score(y_test_labels, y_pred_labels_cnn, average='macro')
f1 = f1_score(y_test_labels, y_pred_labels_cnn, average='macro')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

#plotting the loss
plt.figure(figsize=(8, 5))
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Confusion Matrix for CNN
cm_esn_best = confusion_matrix(y_test_labels, y_pred_labels_cnn)
plt.figure(figsize=(20,20))
sns.heatmap(cm_esn_best, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(49), yticklabels=np.arange(49))
plt.title('Confusion Matrix CNN')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()