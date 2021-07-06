import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(scores, thetas, figure_file, title):
    x = np.arange(len(scores))
    fig, ax1 = plt.subplots()

    ax1.set_title(title)
    ax1.set_xlabel("step")
    ax1.plot(x, scores, "b")
    ax1.set_ylabel("Score", color="b")
    ax1.tick_params("y", colors="b")

    ax2 = ax1.twinx()
    ax2.plot(x, thetas, "r")
    ax2.set_ylabel("Theta", color="r")
    ax2.tick_params("y", colors="r")
    plt.savefig(figure_file)
