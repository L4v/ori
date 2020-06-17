import os
# NOTE(Jovan): CPU ne podrzava AVX, pa ne mogu koristiti tensorflow
os.environ["KERAS_BACKEND"] = "theano"
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

def main():
    data = pd.read_csv("dataset.csv")
    Y = np.array(data.isAlive.values).reshape(-1, 1)
    x1 = data.male.values
    x2 = data.popularity.values
    b1 = data.book1.values
    b2 = data.book2.values
    b3 = data.book3.values
    b4 = data.book4.values
    b5 = data.book5.values
    
    x3 = []
    for i1, i2, i3, i4, i5 in zip(b1, b2, b3, b4, b5):
        x3.append(i1 + i2 + i3 + i4 + i5)

    x4 = data.isNoble.values
    x5 = data.numDeadRelations.values
    x6_tmp = list(data.house.values)
    x6 = []
    for i in x6_tmp:
        if i != i:
            x6.append("None")
        else:
            x6.append(i)

    encoder = LabelEncoder()
    x6 = np.array(x6).reshape(-1, 1)
    encoded = encoder.fit_transform(x6)
    X = []

    for i1, i2, i3, i4, i5, i6 in zip(x1, x2, x3, x4, x5, encoded):
        X.append([i1, i2, i3, i4, i5, i6])
    
    encoder = OneHotEncoder(sparse=False)
    X = np.array(X)

    Y = encoder.fit_transform(Y)
    model = Sequential([
        Dense(6, input_shape=(6,), activation="relu"),
        Dense(10, activation="relu"),
        Dense(2, activation="softmax")
        ])

    optimizer = Adam(lr=0.001)
    model.compile(optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    print(model.summary())
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    model.fit(x_train, y_train, verbose=2, batch_size=20, epochs=200)

    results = model.evaluate(x_test, y_test)
    print("Final test loss: {:4f}".format(results[0]))
    print("Final test accu: {:4f}".format(results[1]))



if __name__ == "__main__":
    main()
