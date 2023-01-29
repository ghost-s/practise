from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
iris = datasets.load_iris()  # Loading the dataset

iris = pd.DataFrame(
    data=np.c_[iris['data'], iris['target']],
    columns=iris['feature_names'] + ['target']
)


setosa = iris[iris.target == 0]
versicolor = iris[iris.target == 1]
virginica = iris[iris.target == 2]

fig, ax = plt.subplots()
fig.set_size_inches(13, 7)  # adjusting the length and width of plot

# lables and scatter points
ax.scatter(setosa['petal length (cm)'], setosa['petal width (cm)'], label="Setosa", facecolor="blue")
ax.scatter(versicolor['petal length (cm)'], versicolor['petal width (cm)'], label="Versicolor", facecolor="green")
ax.scatter(virginica['petal length (cm)'], virginica['petal width (cm)'], label="Virginica", facecolor="red")

ax.set_xlabel("petal length (cm)")
ax.set_ylabel("petal width (cm)")
ax.grid()
ax.set_title("Iris petals")
ax.legend()

X = iris.drop(['target'], axis=1)

X = X.to_numpy()[:, (0, 3)]
y = iris.to_numpy()[:, 4].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5, random_state=42)


# Softmax Function

def Softmax(z):
  exp = np.exp(z)
  for i in range(len(z)):
    exp[i] /= np.sum(exp[i])
  return exp

def fit(X,y, c, epochs, learn_rate):

    # Splitting the number of training examples and features
    (m,n) = X.shape

    # Selecting random weights and bias
    w = np.random.random((n,c))
    b = np.random.random(c)

    loss_arr = []

    # Training
    for epoch in range(epochs):

        # Hypothesis function
        z = X@w + b

        # Computing gradient of loss w.r.t w and b
        grad_for_w = (1/m)*np.dot(X.T,Softmax(z) - OneHot(y, c))
        grad_for_b = (1/m)*np.sum(Softmax(z) - OneHot(y, c))

        # Updating w and b
        w = w - learn_rate * grad_for_w
        b = b - learn_rate * grad_for_b

        # Computing the loss
        loss = -np.mean(np.log(Softmax(z)[np.arange(len(y)), y]))
        loss_arr.append(loss)
        print("Epoch: {} , Loss: {}".format(epoch, loss))

    return w, b, loss_arr


def OneHot(y, c):

    y_encoded = np.zeros((len(y), c))

    y_encoded[np.arange(len(y)), y] = 1
    return y_encoded
# Training the model
w, b, loss = fit(X_train, y_train, c=3, epochs=1000, learn_rate=0.245);


def predict(X, w, b):
    z = X @ w + b
    y_hat = Softmax(z)

    # Returning highest probability class.
    return np.argmax(y_hat, axis=1)


predictions = predict(X_test, w, b)
actual_values = y_test
print(metrics.classification_report(y_test, predictions, digits=3))
print(metrics.confusion_matrix(y_test, predictions))

accuracy = np.sum(actual_values == predictions) / len(actual_values)
print(accuracy)