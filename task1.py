import numpy as np
from sklearn.linear_model import Perceptron as SklearnPerceptron

class Perceptron:
    '''
    perceptron algorithm class
    __init__() initialize the model
    train() trains the model
    predict() predict the class for a new sample

    Attributes:
    - alpha: Learning rate for the perceptron, a positive float.
    - w: Weight vector for the features.
    - b: Bias term.
    '''

    def __init__(self, alpha):
        '''
        Initialize the Perceptron model.

        INPUT:
        - alpha: Learning rate, a float number bigger than 0.
        '''
        if alpha <= 0:
            raise Exception("Sorry, no numbers below or equal to zero. Start again!")

        self.alpha = alpha
        self.w = None  # Initialize weights to None; to be set during training
        self.b = 0     # Bias term initialized to 0

    def train(self, X, y, epochs=100):
        '''
        Train the perceptron on the provided dataset.

        INPUT:
        - X : is a 2D NxD numpy array containing the input features
        - y : is a 1D Nx1 numpy array containing the labels for the corrisponding row of X
        - epochs: Number of iterations over the dataset (default: 100).
        '''
        N, D = X.shape
        self.w = np.zeros(D)  # Initialize weights to zeros

        for epoch in range(epochs):
            for i in range(N):
                # Compute prediction: sign(w^T x + b)
                y_pred = np.sign(np.dot(self.w, X[i]) + self.b)

                # Update weights and bias if prediction is incorrect
                if y_pred != y[i]:
                    self.w += self.alpha * y[i] * X[i]  # Update weights
                    self.b += self.alpha * y[i]         # Update bias

    def predict(self, X_new):
        '''
        Predict the labels for new samples.

        INPUT :
        - X_new : is a MxD numpy array containing the features of new samples whose label has to be predicted
        A

        OUTPUT :
        - y_hat : is a Mx1 numpy array containing the predicted labels for the X_new samples
        '''
        y_hat = np.sign(np.dot(X_new, self.w) + self.b)  # Compute sign(w^T x + b)
        return y_hat



# Define the test dataset (AND logic gate)
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([-1, -1, -1, 1])

# Initialize and train scikit-learn Perceptron
sklearn_perceptron = SklearnPerceptron(alpha=0.1, max_iter=10, tol=None)
sklearn_perceptron.fit(X_train, y_train)

# Predictions using scikit-learn Perceptron
sklearn_predictions = sklearn_perceptron.predict(X_train)

# Validate weights and bias (comparison)
custom_perceptron = Perceptron(alpha=0.5)
custom_perceptron.train(X_train, y_train, epochs=100)
custom_predictions = custom_perceptron.predict(X_train)

custom_weights = custom_perceptron.w
custom_bias = custom_perceptron.b

sklearn_weights = sklearn_perceptron.coef_.flatten()
sklearn_bias = sklearn_perceptron.intercept_.flatten()

print({
    "custom_predictions": custom_predictions.tolist(),
    "custom_weights": custom_weights.tolist(),
    "custom_bias": custom_bias,
    "sklearn_predictions": sklearn_predictions.tolist(),
    "sklearn_weights": sklearn_weights.tolist(),
    "sklearn_bias": sklearn_bias.tolist()
})
