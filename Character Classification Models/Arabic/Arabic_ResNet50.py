# -*- coding: utf-8 -*-
"""Arabic_Resnet50_final.ipynb
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Flatten
from keras.optimizers import Adam


df= pd.read_csv('/content/csvTrainImages 13440x1024.csv')
x_train = df.to_numpy().reshape(13439,32,32,1)
df = pd.read_csv('/content/csvTestImages 3360x1024.csv')
x_test = df.to_numpy().reshape(3359,32,32,1)

df= pd.read_csv('/content/csvTrainLabel 13440x1.csv')
y_train = df.to_numpy()
df= pd.read_csv('/content/csvTestLabel 3360x1.csv')
y_test = df.to_numpy()
y_train.shape

x_train = x_train/255
x_test = x_test/255

x_train=np.dstack([x_train] * 3)
x_train = x_train.reshape(-1, 32,32,3)
x_test=np.dstack([x_test] * 3)
x_test = x_test.reshape(-1, 32,32,3)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_train.shape

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10)

resnet_model = Sequential()

pretrained_model= ResNet50(include_top=False,
                   input_shape=(32,32,3),
                   pooling='avg',
                   classes=29,
                   weights='imagenet')

for layer in pretrained_model.layers:
        layer.trainable=False

resnet_model.add(pretrained_model)
resnet_model.add(Flatten())
resnet_model.add(Dense(512, activation='relu'))
resnet_model.add(Dense(29, activation='softmax'))

resnet_model.summary()

resnet_model.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
history = resnet_model.fit(x_train,y_train,epochs=150,validation_data=(x_val, y_val),batch_size=128)

# Evaluate the resnet model on the test set
y_pred = resnet_model.predict(x_test)
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