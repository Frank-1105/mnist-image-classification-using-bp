import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    exp_scores = np.exp(x)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims= True)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred)) / m

class BPNeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01, decay=0.001, reg_lambda=0.001):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.decay = decay
        self.reg_lambda = reg_lambda
        
        self.W1 = np.random.randn(self.input_size, self.hidden_sizes[0]) * np.sqrt(2.0 / self.input_size)
        self.b1 = np.zeros((1, self.hidden_sizes[0]))
        
        self.W2 = np.random.randn(self.hidden_sizes[0], self.hidden_sizes[1]) * np.sqrt(2.0 / self.hidden_sizes[0])
        self.b2 = np.zeros((1, self.hidden_sizes[1]))
        
        self.W3 = np.random.randn(self.hidden_sizes[1], self.output_size) * np.sqrt(2.0 / self.hidden_sizes[1])
        self.b3 = np.zeros((1, self.output_size))

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = relu(self.Z1)

        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = relu(self.Z2)

        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = softmax(self.Z3)

        return self.A3

    def backward(self, X, y, output):
        m = y.shape[0]
        
        dZ3 = output - y
        dW3 = np.dot(self.A2.T, dZ3) / m + self.reg_lambda * self.W3 / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m

        dA2 = np.dot(dZ3, self.W3.T)
        dZ2 = dA2 * relu_derivative(self.A2)
        dW2 = np.dot(self.A1.T, dZ2) / m + self.reg_lambda * self.W2 / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * relu_derivative(self.A1)
        dW1 = np.dot(X.T, dZ1) / m + self.reg_lambda * self.W1 / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.W1 -= self.learning_rate * dW1
        self.W2 -= self.learning_rate * dW2
        self.W3 -= self.learning_rate * dW3
        self.b1 -= self.learning_rate * db1
        self.b2 -= self.learning_rate * db2
        self.b3 -= self.learning_rate * db3

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            output = self.forward(X)
            loss = mean_squared_error(y, output)
            self.backward(X, y, output)
            self.learning_rate *= (1 / (1 + self.decay * epoch))  

            if epoch % 1 == 0:
                print(f'Epoch {epoch}/{epochs} - Loss: {loss}')

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"].astype(np.int32)

X = X / 255.0

encoder = OneHotEncoder(sparse_output=False)
y_one_hot = encoder.fit_transform(np.array(y).reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
print("划分成功...")
nn = BPNeuralNetwork(input_size=784, hidden_sizes=[128, 64], output_size=10, learning_rate=0.036, decay=0, reg_lambda=0.005)
print("开始训练...")
nn.train(X_train, y_train, epochs=2000)

y_pred = nn.predict(X_test)

y_test_labels = np.argmax(y_test, axis=1)


accuracy = accuracy_score(y_test_labels, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
