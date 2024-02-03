# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Define data directories
training_dir = '/content/mnist_train.csv'
test_dir = '/content/mnist_test.csv'

# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    # Read CSV file
    df = pd.read_csv(file_path)

    # Separate images and labels
    x = df.drop('label', axis=1).to_numpy()
    y = df['label'].to_numpy()

    # Reshape images into 28x28 pixels and scale pixel values
    x = x.reshape(-1, 28, 28, 1) / 255

    # One-hot encode labels
    y = tf.keras.utils.to_categorical(y)

    return x, y

# Load and preprocess training and test data
x_train, y_train = load_and_preprocess_data(training_dir)
x_test, y_test = load_and_preprocess_data(test_dir)

# Create CNN model
model = Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

# Display model summary
model.summary()

# Compile the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=128)

# Make predictions on the test set
y_pred = model.predict(x_test)

# Convert probabilities to class labels
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Print classification report
print("Classification Report using CNN:")
print(classification_report(y_test_labels, y_pred_labels))

# Calculate and print metrics
accuracy = accuracy_score(y_test_labels, y_pred_labels)
precision = precision_score(y_test_labels, y_pred_labels, average='macro')
recall = recall_score(y_test_labels, y_pred_labels, average='macro')
f1 = f1_score(y_test_labels, y_pred_labels, average='macro')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Plot confusion matrix
cm = confusion_matrix(y_test_labels, y_pred_labels)
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(10), yticklabels=np.arange(10))
plt.title('Confusion Matrix CNN')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()