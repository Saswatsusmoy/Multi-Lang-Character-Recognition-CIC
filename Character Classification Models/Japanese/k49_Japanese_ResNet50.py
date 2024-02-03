# -*- coding: utf-8 -*-
"""K49_JAPANESE_Resnet50_final.ipynb
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import keras
from keras import layers
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from keras import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam

train_imgs_path = '/content/k49-train-imgs.npz'
train_labels_path = '/content/k49-train-labels.npz'
test_imgs_path = '/content/k49-test-imgs.npz'
test_labels_path = '/content/k49-test-labels.npz'

# Data loading
x_train = np.load(train_imgs_path)['arr_0']
y_train= np.load(train_labels_path)['arr_0']
x_test= np.load(test_imgs_path)['arr_0']
y_test= np.load(test_labels_path)['arr_0']

x_train = x_train.reshape(232365,28,28,1)

x_test = x_test.reshape(38547,28,28,1)

x_train=np.dstack([x_train] * 3)
x_train = x_train.reshape(-1, 28,28,3)
x_test=np.dstack([x_test] * 3)
x_test = x_test.reshape(-1, 28,28,3)



x_train_padded = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='constant', constant_values=0)
x_test_padded = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='constant', constant_values=0)
x_train_padded.shape

x_train, x_val, y_train, y_val = train_test_split(x_train_padded, y_train, test_size=0.10)

y_train = keras.utils.to_categorical(y_train)
y_test= keras.utils.to_categorical(y_test)
y_val=keras.utils.to_categorical(y_val)
y_test.shape

# Creating Resnet model
resnet_model = Sequential()

pretrained_model= ResNet50(include_top=False,
                   input_shape=(32,32,3),
                   pooling='avg',
                   classes=49,
                   weights='imagenet')

for layer in pretrained_model.layers:
        layer.trainable=False

resnet_model.add(pretrained_model)
resnet_model.add(Flatten())
resnet_model.add(Dense(512, activation='relu'))
resnet_model.add(Dense(49, activation='softmax'))

resnet_model.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

# Let's look at our model
resnet_model.summary()

history = resnet_model.fit(x_train, y_train, epochs=5,validation_data=(x_val, y_val), batch_size=128)

# Evaluate the resnet model on the test set
y_pred = resnet_model.predict(x_test_padded)
y_pred_labels_cnn = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Classification report for Resnet model on test data
accuracy_resnet_best = accuracy_score(y_test_labels, y_pred_labels_cnn)
print(f"Accuracy on the test set using Resnet : {accuracy_resnet_best * 100:.2f}%")
print("Classification Report using Resnet:")
print(classification_report(y_test_labels, y_pred_labels_cnn))



precision_resnet = precision_score(y_test_labels, y_pred_labels_cnn, average='weighted')
recall_resnet = recall_score(y_test_labels, y_pred_labels_cnn, average='weighted')
f1_resnet = f1_score(y_test_labels, y_pred_labels_cnn, average='weighted')

print(f"Accuracy on the test set using Resnet: {accuracy_resnet_best * 100:.2f}%")
print(f"Precision on the test set using Resnet: {precision_resnet * 100:.2f}%")
print(f"Recall on the test set using Resnet: {recall_resnet * 100:.2f}%")
print(f"F1 Score on the test set using Resnet: {f1_resnet * 100:.2f}%")

plt.figure(figsize=(8, 5))
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

cm_esn_best = confusion_matrix(y_test_labels, y_pred_labels_cnn)
plt.figure(figsize=(12,12))
sns.heatmap(cm_esn_best, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(28), yticklabels=np.arange(28))
plt.title('Confusion Matrix Resnet')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()