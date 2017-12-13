import numpy as np


class ColorClass:
    def __init__(self):
        self.samples = []

    def length(self):
        return len(self.samples)

    def add_data_point(self, rgb_data):
        self.samples.append(map(float, rgb_data))

    def get_samples_r(self):
        return [self.samples[i][0] for i in range(len(self.samples))]

    def get_samples_g(self):
        return [self.samples[i][1] for i in range(len(self.samples))]

    def get_samples_b(self):
        return [self.samples[i][2] for i in range(len(self.samples))]

    def __get_samples_r_g_b(self):
        return [self.get_samples_r(), self.get_samples_g(), self.get_samples_b()]

    def add_samples(self, image, mask):
        ny, nx, nz = np.shape(image)
        for r in range(ny):
            for c in range(nx):
                if mask[r][c]:
                    self.add_data_point(image[r][c])

    def __mean(self, samples):
        # private method to calculate the mean of samples
        s = 0
        for i in samples:
            s = s + i
        return s / self.length()

    def mean_rgb(self):
        return [self.__mean(self.get_samples_r()), self.__mean(self.get_samples_g()), self.__mean(self.get_samples_b())]

    def __covariance_helper(self, i, j):
        # find cov at (i,j) entry
        # = E{XiXj} - mui*muj
        X = self.__get_samples_r_g_b()
        mu = self.mean_rgb()
        Xi = X[i]
        Xj = X[j]
        n = self.length()
        s = 0
        for m in range(n):
            s = s + Xi[m]*Xj[m]
        return s / n - mu[i]*mu[j]

    def covariance(self):
        return [[self.__covariance_helper(r, c) for c in range(3)] for r in range(3)]
