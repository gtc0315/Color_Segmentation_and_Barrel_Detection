import numpy as np


class LinearRegressionDistanceModel:
    def __init__(self):
        self.X = []
        self.y = []
        self.A = []

    def add_sample(self, width, height, dist):
        self.X.append([width, height, 1])
        self.y.append(dist)

    def calculate_estimator(self):
        X = np.matrix(self.X)
        y = np.transpose(np.matrix(self.y))
        self.A = np.linalg.inv(np.transpose(X)*X)*np.transpose(X)*y

    def estimate_dist(self, width, height):
        if len(self.A) == 0:
            self.calculate_estimator()
        return np.asscalar([width, height, 1]*self.A)

    def __mean(self, ratios):
        s = 0
        for i in ratios:
            s = s + i
        return s / len(ratios)

    def __variance(self,ratios, mean):
        s = 0
        for i in ratios:
            s = s + (i-mean)**2
        return s / len(ratios)

    def ratio_statistics(self):
        width_height_ratio = [i[0] * 1.0 / i[1] for i in self.X]
        ratio_mean = self.__mean(width_height_ratio)
        ratio_std = np.sqrt(self.__variance(width_height_ratio, ratio_mean))
        return ratio_mean, ratio_std

