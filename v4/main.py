import pandas as pd
import random
import matplotlib.pyplot as plt

def main():
    data = pd.read_csv("data/skincancer.csv", delimiter=',', index_col=0)
    # y = Mort
    # x = Lat
    x = data.Lat.values
    y = data.Mort.values
    lin_reg = LinearRegression(x, y)
    hawaii = lin_reg.predict(20)
    print(hawaii)
    k_means = KMeans()

    lat = data.Lat.values
    lon = data.Long.values
    for i, j in zip(lat, lon):
        k_means.points.append(Point(i, j))
    k_means.split(4, 0.01)
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    for cluster in k_means._clusters:
        ax.scatter(cluster.elements[:].x, cluster.elements[:].y)
    plt.show()

class LinearRegression:
    def __init__(self, x, y):
        self._a = 0
        self._b = 0
        self._regression(x, y)

    def _regression(self, x, y):
        n = len(x)
        xy_sum = 0
        for xi, yi in zip(x, y):
            xy_sum += xi * yi
        x_sum = sum(x)
        y_sum = sum(y)
        xx_sum = 0
        for xi in x:
            xx_sum += xi * xi

        self._a = n * (xy_sum) - x_sum * y_sum
        self._a = self._a / (n*xx_sum - x_sum * x_sum)
        self._b = 1/n * (y_sum - self._a * x_sum)

    def predict(self, x):
        return x * self._a + self._b

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Cluster:
    def __init__(self):
        self.center = Point(0, 0)
        self.elements = []

    def distance(self, point):
        return abs(self.center.x - point.x) + abs(self.center.y - point.y)

    def move_center(self):
        x_sum = 0
        y_sum = 0
        for p in self.elements:
            x_sum += p.x
            y_sum += p.y

        n = len(self.elements)
        if n <= 0: return 0
        self.center = Point(x_sum / n, y_sum / n)
        return self.distance(self.center)

    
class KMeans:
    def __init__(self):
        self.points = []
        self._clusters = []
        self._groups = 0

    def split(self, groups, errT):
        self._groups = groups
        if groups <= 0: return
        # Init
        r = random.SystemRandom()
        for i in range(groups):
            mu = r.randint(0, len(self.points))
            cluster = Cluster()
            cluster.center = self.points[mu]
            self._clusters.append(cluster)
        # Iterative calc
        for i in range(1000):
            for cluster in self._clusters:
                cluster.elements = []

            for point in self.points:
                closest = 0
                for j in range(self._groups):
                    if (self._clusters[closest].distance(point) 
                            > self._clusters[j].distance(point)):
                        closest = j
                    self._clusters[closest].elements.append(point)

            err = 0
            for j in range(self._groups):
                err += self._clusters[j].move_center()

            if err < errT: break



if __name__ == "__main__":
    main()
