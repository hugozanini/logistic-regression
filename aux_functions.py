import matplotlib.pyplot as plt
import numpy as np

def sigmoid(z: float) -> (float):
    return 1.0/(1 + np.exp(-z))

def loss(y: float, y_hat: float) -> (float):
    loss = -np.mean(y*(np.log(y_hat)) - (1-y)*np.log(1-y_hat))
    return loss

def gradients(X: list, y: list, y_hat: list) -> (float, float):
    '''
    X --> Input.
    y --> true/target value.
    y_hat --> hypothesis/predictions.
    w --> weights (parameter).
    b --> bias (parameter).
    '''
    m = X.shape[0]

    # Gradient of loss w.r.t weights.
    dw = (1/m)*np.dot(X.T, (y_hat - y))

    # Gradient of loss w.r.t bias.
    db = (1/m)*np.sum((y_hat - y))
    return dw, db

def plot_data(X: np.array, y: np.array) -> (None):
    '''
    X --> Inputs
    y --> class
    '''
    fig = plt.figure(figsize=(10,8))
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "g^")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs")
    plt.xlim([X.min(), X.max()])
    plt.ylim([X.min(), X.max()])
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title('Dataset')

def plot_decision_boundary(X: np.array, y: np.array, w: float, b: float) -> (None):
    '''
    X --> Inputs
    w --> weights
    b --> bias
    The Line is y=mx+c
    So, Equate mx+c = w.X + b
    Solving we find m and c
    '''
    x1 = [min(X[:,0]), max(X[:,0])]
    m = -w[0]/w[1]
    c = -b/w[1]
    x2 = m*x1 + c

    fig = plt.figure(figsize=(10,8))
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "g^")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs")
    plt.xlim([X.min(), X.max()])
    plt.ylim([X.min(), X.max()])
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.title('Decision Boundary')
    plt.plot(x1, x2, 'y-')

def normalize(X: np.array) -> (np.array):
    '''
    X --> Input.
    m-> number of training examples
    n-> number of features
    m, n = X.shape
    '''
    m, n = X.shape
    # Normalizing all the n features of X.
    for i in range(n):
        X = (X - X.mean(axis=0))/X.std(axis=0)

    return X

def train(X: np.array, y: np.array, bs: int, epochs: int, lr: float) -> (np.array, float, list):
    '''
    X --> Input.
    y --> true/target value.
    bs --> Batch Size.
    epochs --> Number of iterations.
    lr --> Learning rate.

    m-> number of training examples
    n-> number of features
    '''
    m, n = X.shape

    # Initializing weights and bias to zeros.
    w = np.zeros((n,1))
    b = 0

    # Reshaping y.
    y = y.reshape(m,1)

    # Normalizing the inputs.
    x = normalize(X)

    # Empty list to store losses.
    losses = []

    # Training loop.
    for epoch in range(epochs):
        for i in range((m-1)//bs + 1):

            # Defining batches. SGD.
            start_i = i*bs
            end_i = start_i + bs
            xb = X[start_i:end_i]
            yb = y[start_i:end_i]

            # Calculating hypothesis/prediction.
            y_hat = sigmoid(np.dot(xb, w) + b)

            # Getting the gradients of loss w.r.t parameters.
            dw, db = gradients(xb, yb, y_hat)

            # Updating the parameters.
            w -= lr*dw
            b -= lr*db

        # Calculating loss and appending it in the list.
        l = loss(y, sigmoid(np.dot(X, w) + b))
        losses.append(l)

    return w, b, losses

def predict(X: np.array) -> (np.array):
    '''
    X --> Input.
    '''

    # Normalizing the inputs.
    x = normalize(X)

    # Calculating presictions/y_hat.
    preds = sigmoid(np.dot(X, w) + b)

    # Empty List to store predictions.
    pred_class = []
    # if y_hat >= 0.5 --> round up to 1
    # if y_hat < 0.5 --> round up to 1
    pred_class = [1 if i > 0.5 else 0 for i in preds]

    return np.array(pred_class)