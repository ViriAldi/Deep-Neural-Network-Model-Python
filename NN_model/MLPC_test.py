from NN_model.model import *
from keras.datasets import fashion_mnist
from sklearn.metrics import precision_score
import PIL.Image
import matplotlib.pyplot as plt


def load_data():
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = np.array([train_images[i] for i in range(60000) if train_labels[i] < 2])
    train_labels = np.array([train_labels[i] for i in range(60000) if train_labels[i] < 2])

    test_images = np.array([test_images[i] for i in range(10000) if test_labels[i] < 2])
    test_labels = np.array([test_labels[i] for i in range(10000) if test_labels[i] < 2])

    train_images = train_images.reshape(12000, 28 * 28).T
    test_images = test_images.reshape(2000, 28 * 28).T

    X_train = train_images
    y_train = train_labels.reshape((1, 12000))
    X_test = test_images
    y_test = test_labels.reshape((1, 2000))

    # X_train = X_train - np.mean(X_train)
    # X_test = X_test - np.mean(X_test)

    X_train = X_train / (np.sum(X_train**2, axis=0, keepdims=True) / X_train.shape[1])
    X_test = X_test / (np.sum(X_test**2, axis=0, keepdims=True) / X_test.shape[1])

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    model = MultiLayerPerceptronClassifier([10, 5], [tanh, tanh, tanh, tanh], num_iter=500, learning_rate=0.1, alpha=0.)
    model.fit(X_train, y_train)

    return model


if __name__ == "__main__":
    np.random.seed(0)
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    curve = model.loss_curve

    plt.plot(curve)
    plt.show()

    pred_train = np.round(model.predict(X_train)).astype(int)[0]
    pred_test = np.round(model.predict(X_test)).astype(int)[0]

    print(precision_score(y_train.reshape(12000, 1), pred_train))
    print(precision_score(y_test.reshape(2000, 1), pred_test))

    # img = PIL.Image.open("../data/images/trousers.jpg")
    # img = img.resize((28, 28))
    #
    # if np.array(img).shape == (28, 28, 3):
    #     img = (np.sum(np.array(img), axis=2) // 3).reshape((28*28, 1))
    # else:
    #     img = np.array(img).reshape((28*28, 1))
    #
    # predict = model.predict(img)
    # final = np.round(predict).astype(int)[0][0]
    #
    # if final == 0:
    #     print("T-SHIRT")
    #
    # else:
    #     print("TROUSERS")
