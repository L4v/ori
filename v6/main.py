import os
# NOTE(Jovan): CPU ne podrzava AVX, pa ne mogu koristiti tensorflow
os.environ["KERAS_BACKEND"] = "theano"
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from sklearn.preprocessing import OneHotEncoder

import pandas as pd
import numpy as np


def main():

    test_data = pd.read_csv("occupancy_test.csv")
    train_data = pd.read_csv("occupancy_train.csv")
    x_test = []
    x_train = []
    for i1, i2, i3, i4 in zip(test_data.Humidity.values, test_data.CO2.values, test_data.Light.values, test_data.HumidityRatio.values):
        x_test.append([i1, i2, i3, i4])

    for i1, i2, i3, i4 in zip(train_data.Humidity.values, train_data.CO2.values, train_data.Light.values, train_data.HumidityRatio.values):
        x_train.append([i1, i2, i3, i4])

    x_test = np.array(x_test)
    x_train = np.array(x_train)

    y_test = np.array(test_data.Occupancy.values).reshape(-1, 1)
    y_train = np.array(train_data.Occupancy.values).reshape(-1, 1)

    encoder = OneHotEncoder(sparse=False)
    y_test = encoder.fit_transform(y_test)
    y_train = encoder.fit_transform(y_train)

    model = Sequential([
        Dense(4, input_shape=(4,), activation="relu"),
        Dense(4, activation="relu"),
        Dense(2, activation="softmax")
        ])

    optimizer = Adam(lr=0.001)
    model.compile(optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    print(model.summary())

    model.fit(x_train, y_train, verbose=2, batch_size=100, epochs=100)

    results = model.evaluate(x_test, y_test)
    print("Final test loss: {:4f}".format(results[0]))
    print("Final test accu: {:4f}".format(results[1]))


if __name__ == "__main__":
    main()
