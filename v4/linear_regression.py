# Linear regression
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
