import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d


def plot_spiketrain(spiketrain, sigma=10, ax=None):
    if ax is None:
        ax = plt
    ax.plot(spiketrain.index.total_seconds(), gaussian_filter1d(spiketrain, sigma))
    ax.set_xlim((spiketrain.index[0].total_seconds(),
                 spiketrain.index[-1].total_seconds()))


def plot_targets(spiketrain, targets, bins, ax=None):
    if ax is None:
        ax = plt
    assert(len(targets) % 2 ==0)
    ax.plot(spiketrain.index.total_seconds(), targets - len(bins) // 2)
    ax.set_xlim((spiketrain.index[0].total_seconds(),
                 spiketrain.index[-1].total_seconds()))
