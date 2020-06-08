from NN_model.model import *
from keras.datasets import fashion_mnist
import PIL.Image


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

    X_train = X_train / np.sum(X_train ** 2, axis=1, keepdims=True) ** 0.5
    X_test = X_test / np.sum(X_test ** 2, axis=1, keepdims=True) ** 0.5

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    model = MultiLayerPerceptronClassifier([10], [tanh, tanh], num_iter=100, learning_rate=0.5)
    model.fit(X_train, y_train)

    return model


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)

    img = PIL.Image.open("../data/images/trousers.jpg")
    img = img.resize((28, 28))

    if np.array(img).shape == (28, 28, 3):
        img = (np.sum(np.array(img), axis=2) // 3).reshape((28*28, 1))
    else:
        img = np.array(img).reshape((28*28, 1))

    predict = model.predict(img)
    final = np.round(predict).astype(int)[0][0]

    if final == 0:
        print("T-SHIRT")

    else:
        print("TROUSERS")
