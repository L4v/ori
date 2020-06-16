from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

def main():
    data = pd.read_csv('dataset.csv')
    x1 = data.book1.values
    x2 = data.book2.values
    x3 = data.book3.values
    x4 = data.book4.values
    x5 = data.book5.values
    X = []
    for i1, i2, i3, i4, i5 in zip(x1, x2, x3, x4, x5):
        if i1 + i2 + i3 + i4 + i5 >= 3:
            X.append([i1, i2, i3, i4, i5])

    df = pd.DataFrame()
    df['data'] = X
    X = np.array(X)
    km = KMeans(n_clusters=3)
    km.fit(X)
    df['cluster'] = km.labels_

    no_c0 = len(df[df.cluster == 0])
    no_c1 = len(df[df.cluster == 1])
    no_c2 = len(df[df.cluster == 2])
    
    # NOTE(Jovan): Bar 3 knjige
    c0 = 0
    c1 = 0
    c2 = 0

    for char in df[df.cluster==0].data:
        s = sum(list(char))
        if s == 5:
            c0 += 1
    
    for char in df[df.cluster==1].data:
        s = sum(list(char))
        if s == 5:
            c1 += 1

    for char in df[df.cluster==2].data:
        s = sum(list(char))
        if s == 5:
            c2 += 1

    print(c0, c1, c2)
    print("Cluster 0: ", c0 / no_c0 * 100, "%")
    print("Cluster 1: ", c1 / no_c1 * 100, "%")
    print("Cluster 2: ", c2 / no_c2 * 100, "%")
    print("Cluster sum: ", no_c0 + no_c1 + no_c2, " Original data size:", len(x1))

if __name__ == "__main__":
    main()
