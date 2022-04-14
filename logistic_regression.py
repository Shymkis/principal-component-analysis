from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import time
import warnings

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def train_model(X, Y, epochs=200, eta=.2, lmbda=0, tol=.00001):
    warnings.filterwarnings("ignore")
    X = np.hstack((np.ones((X.shape[0], 1)), X)) # Add intercept
    n_labels = len(np.unique(Y))
    w = np.zeros((n_labels, X.shape[1]))
    # One vs. all
    for i in range(n_labels):
        for _ in range(epochs):
            diff = (Y == i) - sigmoid(np.dot(X, w[i]))
            step = -eta*lmbda*w[i] + eta*np.dot(diff, X)
            w[i] += step
            if (abs(step) < tol).all():
                break
    return w

def predict(lr_model, X):
    X = np.hstack((np.ones((X.shape[0], 1)), X)) # Add intercept
    return np.argmax(X @ lr_model.T, axis=1)

def prediction_accuracy(labels, predictions):
    return np.mean(labels == predictions) * 100

def main(n_comps=None):
    # Load data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    # Standardize
    sc_model = StandardScaler().fit(x_train)
    x_train = sc_model.transform(x_train)
    x_test = sc_model.transform(x_test)

    # Optional PCA
    if n_comps:
        start = time.time()
        pca_model = PCA(n_comps).fit(x_train)
        x_train = pca_model.transform(x_train)
        x_test = pca_model.transform(x_test)
        pca_time = round(time.time() - start, 2)
        print("PCA time:", pca_time)

    # Training
    start = time.time()
    lr_model = train_model(x_train, y_train)
    training_time = round(time.time() - start, 2)
    print("Training time:", training_time)

    # Display model
    # for i in range(10):
    #     plt.subplot(2, 5, 1 + i)
    #     plt.imshow(lr_model[i][1:].reshape((28, 28)), cmap=plt.get_cmap('gray'))
    #     plt.axis("off")
    # plt.show()

    # Make predictions on training set and testing set independently
    train_predictions = predict(lr_model, x_train)
    test_predictions = predict(lr_model, x_test)

    # Calculate accuracies
    train_accuracy = round(prediction_accuracy(y_train, train_predictions), 2)
    test_accuracy = round(prediction_accuracy(y_test, test_predictions), 2)
    print("Train accuracy:", train_accuracy)
    print("Test accuracy:", test_accuracy)

if __name__ == "__main__":
    main()
    main(n_comps=150)
    main(n_comps=100)
    main(n_comps=50)
