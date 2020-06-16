# KMeans
import random

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Cluster:
    def __init__(self):
        self.center = Point(0, 0)
        self.points = []

    def distance(self, point):
        return abs(self.center.x - point.x) + abs(self.center.y - point.y)

    def move_center(self):
        x_sum = 0
        y_sum = 0
        for p in self.points:
            x_sum += p.x
            y_sum += p.y

        n = len(self.points)
        if n <= 0: return 0
        self.center = Point(int(x_sum / n), int(y_sum / n))
        return self.distance(self.center)

    
class KMeans:
    def __init__(self):
        self.points = []
        self._clusters = []
        self._no_clusters = 0

    def split(self, no_clusters, errT):
        self._no_clusters = no_clusters
        if no_clusters <= 0: return
        # Init
        r = random.SystemRandom()
        for i in range(no_clusters):
            mu = r.randint(0, len(self.points) - 1)
            cluster = Cluster()
            cluster.center = self.points[mu]
            self._clusters.append(cluster)
        # Iterative calc
        for i in range(1000):
            for cluster in self._clusters:
                cluster.points = []

            for point in self.points:
                closest = 0
                for j in range(self._no_clusters):
                    if (self._clusters[closest].distance(point) 
                            > self._clusters[j].distance(point)):
                        closest = j
                    self._clusters[closest].points.append(point)

            err = 0
            for j in range(self._no_clusters):
                err += self._clusters[j].move_center()

            if err < errT: break
