import numpy as np


class OUNoise:
    def __init__(self, shape, seed, mu=0.0, theta=0.01, theta_incr=0.01, sigma=0.001):
        self.mu = mu * np.ones(shape)
        self.theta_incr = theta_incr
        self.theta = theta
        self.sigma = sigma
        self.seed = np.random.seed(seed)
        self.reset(incr_theta=False)

    def reset(self, incr_theta=True):
        # Start with high noise randomly at +- 1
        self.state = np.sign(2 * np.random.random(self.mu.shape) - 1)
        self.i = 0
        if incr_theta:
            self.theta = min(self.theta + self.theta_incr, 1.0)

    def sample(self):
        self.i += 1
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * (
            2 * np.random.random(self.state.shape) - 1
        )
        self.state = x + dx
        return self.state
