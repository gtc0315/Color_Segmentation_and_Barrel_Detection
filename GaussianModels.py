import numpy as np


class GaussianModels:

    def __init__(self, mu, cov, theta):
        self.mu = mu
        self.cov = cov
        self.theta = theta
        self.p_y = [np.log(theta[i]) for i in range(4)]
        self.inv_cov = [np.linalg.inv(np.matrix(cov[i])) for i in range(4)]
        self.det_cov = [np.linalg.det(np.matrix(cov[i])) for i in range(4)]
        self.log_phi_constant = [-0.5 * np.log(2 * np.pi * self.det_cov[i]) for i in range(4)]

    def log_p_joint_x_y(self, x, y):
        mu = self.mu[y]
        temp = np.matrix(x - mu) * np.matrix(self.inv_cov[y]) * np.matrix(np.transpose(x - mu))
        return self.p_y[y] + self.log_phi_constant[y] - 0.5 * np.diag(temp)

    def image_classify(self, x):
        x_reshaped = np.reshape(x, (-1, np.shape(x)[-1]))
        p = [self.log_p_joint_x_y(x_reshaped, y) for y in range(4)]
        img_data = np.argmax(p, axis=0)
        return np.reshape(img_data,np.shape(x)[0:-1])
