import numpy as np


class OUNoise:
    def __init__(
        self, shape, seed=None, mu=0.0, theta=0.01, theta_incr=0.01, sigma=0.001
    ):
        self.mu = mu * np.ones(shape)
        self.theta_incr = theta_incr
        self.theta = theta
        self.sigma = sigma
        if seed:
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    noise = OUNoise(1, sigma=0.01, theta=0.10)
    x = np.arange(50)
    y1 = [noise.sample() for _ in x]
    noise = OUNoise(1, sigma=0.01, theta=0.5)
    y2 = [noise.sample() for _ in x]
    noise = OUNoise(1, sigma=0.01, theta=0.90)
    noise.theta = 0.95
    y3 = [noise.sample() for _ in x]

    plt.plot(x, y1, "r", label="theta=0.10")
    plt.plot(x, y2, "g", label="theta=0.50")
    plt.plot(x, y3, "b", label="theta=0.90")
    plt.legend()
    plt.title("Effects of increasing theta on OUNoise (mu=0, sigma=0.01)")
    plt.ylabel("Noise")
    plt.xlabel("Step")
    plt.show()
