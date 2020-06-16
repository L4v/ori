import pandas as pd
import matplotlib.pyplot as plt

from linear_regression import LinearRegression
from kmeans import KMeans, Point

# NOTE(Jovan): Rucno napravljen KMeans, moze i sa sklearn
def main():
    # NOTE(Jovan): Load data
    data = pd.read_csv("data/skincancer.csv", delimiter=',', index_col=0)
    mort = data.Mort.values
    lat = data.Lat.values
    lon = data.Long.values

    # NOTE(Jovan): Init LinearRegression and predict
    lin_reg = LinearRegression(lat, mort)
    hawaii = lin_reg.predict(20)
    print("Prediction for hawaii[lat=20]:", hawaii)

    # NOTE(Jovan): Init KMeans and add lat and long points
    k_means = KMeans()
    for i, j in zip(lat, lon):
        k_means.points.append(Point(i, j))
    k_means.split(2, 0.01)

    # NOTE(Jovan): Plot clusters
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    # NOTE(Jovan): First clusters
    for p in k_means._clusters[0].points:
        ax.scatter(p.x, p.y, c="#ff0000")
    # NOTE(Jovan): Second clusters
    for p in k_means._clusters[1].points:
        ax.scatter(p.x, p.y, c="#00ff00")

    # NOTE(Jovan): Plot cluster centers
    center1 = k_means._clusters[0].center
    center2 = k_means._clusters[1].center
    ax.scatter(center1.x, center1.y, marker="P", c="#ff0000")
    ax.scatter(center2.x, center2.y, marker="P", c="#00ff00")
    plt.show()

if __name__ == "__main__":
    main()
