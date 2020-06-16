import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


def main():
    data = pd.read_csv("skincancer.csv")
    lat = data.Lat.values
    lon = data.Long.values
    X = []
    for i, j in zip(lat, lon):
        X.append([i, j])

    cluster_map = pd.DataFrame()
    cluster_map["data"] = X
    X = np.array(X)
    km = KMeans(n_clusters = 2)
    km.fit(X)

    cluster_map["cluster"] = km.labels_
    print(cluster_map)

    plt.scatter(X[:, 0], X[:, 1], c=km.labels_)
    plt.show()


if __name__ == "__main__":
    main()
