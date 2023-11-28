import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
)
from sklearn.preprocessing import label_binarize

# Load MNIST dataset
train_images = idx2numpy.convert_from_file('path/to/train-images.idx3-ubyte')
train_labels = idx2numpy.convert_from_file('path/to/train-labels.idx1-ubyte')
test_images = idx2numpy.convert_from_file('path/to/t10k-images.idx3-ubyte')
test_labels = idx2numpy.convert_from_file('path/to/t10k-labels.idx1-ubyte')

X_train = train_images.reshape(train_images.shape[0], -1) / 255.0
X_test = test_images.reshape(test_images.shape[0], -1) / 255.0
y_train = np.eye(10)[train_labels]  # One-hot encode labels
y_test = np.eye(10)[test_labels]

class EuSN:
    def __init__(self, N, X, omega_r, omega_x, omega_b, epsilon=0.01, gamma=0.01):
        self.N = N
        self.X = X
        self.omega_r = omega_r
        self.omega_x = omega_x
        self.omega_b = omega_b
        self.epsilon = epsilon
        self.gamma = gamma

        # Initialize weights and biases
        self.Wh = self.initialize_weights()
        self.Wx = np.random.uniform(-omega_x, omega_x, (N, X))
        self.b = np.random.uniform(-omega_b, omega_b, (N,))

        # Initialize reservoir state
        self.h = np.zeros(N)

    def initialize_weights(self):
        W = np.random.uniform(-self.omega_r, self.omega_r, (self.N, self.N))
        return W - np.transpose(W)

    def forward(self, x):
        self.h = self.h + self.epsilon * np.tanh(
            (self.Wh - self.gamma * np.identity(self.N)) @ self.h + self.Wx @ x + self.b
        )
        return self.h

    def train_readout_layer(self, inputs, targets):
        H = np.zeros((len(inputs), self.N))
        for i, x in enumerate(inputs):
            H[i, :] = self.forward(x)

        H_with_bias = np.column_stack((H, np.ones((len(inputs), 1))))
        self.readout_weights = np.linalg.lstsq(H_with_bias, targets, rcond=None)[0]

    def predict(self, inputs):
        H = np.zeros((len(inputs), self.N))
        for i, x in enumerate(inputs):
            H[i, :] = self.forward(x)

        H_with_bias = np.column_stack((H, np.ones((len(inputs), 1))))
        predictions = np.dot(H_with_bias, self.readout_weights)
        return predictions
    
    def evaluate(self,inputs,targets):
        predictions = self.predict(inputs)
        y_true = np.argmax(targets, axis=1)
        y_pred = np.argmax(predictions, axis=1)

        accuracy = accuracy_score(y_true, y_pred)
        confusion_mat = confusion_matrix(y_true, y_pred)
        classification_rep = classification_report(y_true, y_pred)

        # Precision-Recall curve
        y_bin = label_binarize(y_true, classes=np.arange(10))
        precision, recall, _ = precision_recall_curve(y_bin.ravel(), predictions.ravel())
        pr_auc = auc(recall, precision)

        # Additional metrics
        weighted_avg_precision = precision_score(y_true, y_pred, average='weighted')
        weighted_avg_recall = recall_score(y_true, y_pred, average='weighted')
        weighted_avg_f1 = f1_score(y_true, y_pred, average='weighted')
        mcc = matthews_corrcoef(y_true, y_pred)

        return (
            accuracy,
            confusion_mat,
            classification_rep,
            precision,
            recall,
            pr_auc,
            weighted_avg_precision,
            weighted_avg_recall,
            weighted_avg_f1,
            mcc,
        )

# Example usage
N = 100
X_dim = X_train.shape[1]
omega_r = 1.0
omega_x = 1.0
omega_b = 1.0
eusrnn = EuSN(N, X_dim, omega_r, omega_x, omega_b)
# Train readout layer
eusrnn.train_readout_layer(X_train, y_train)
predictions = eusrnn.predict(X_test)

# Evaluate on test set
(
        accuracy,
    confusion_mat,
    classification_rep,
    precision,
    recall,
    pr_auc,
    weighted_avg_precision,
    weighted_avg_recall,
    weighted_avg_f1,
    mcc,
) = eusrnn.evaluate(X_test, y_test)

# Print results
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_mat)
print("\nClassification Report:")
print(classification_rep)
print(f"\nPrecision-Recall AUC: {pr_auc:.2f}")
print(f"Weighted Avg. Precision: {weighted_avg_precision:.2f}")
print(f"Weighted Avg. Recall: {weighted_avg_recall:.2f}")
print(f"Weighted Avg. F1-score: {weighted_avg_f1:.2f}")
print(f"Matthews Correlation Coefficient: {mcc:.2f}")

# Plot Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR Curve (AUC = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower right')
plt.show()