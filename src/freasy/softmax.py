import numpy as np


def softmax(sources_distribution, temperature=1.0, power=1.0):
    sources = [s for s, _ in sources_distribution]
    weights = np.array([w for _, w in sources_distribution])

    e = np.exp(weights / temperature)
    softmaxed = e / np.sum(e)
    # powered = np.power(softmaxed, [power])

    sources_distribution_softmaxed = list(zip(sources, softmaxed))

    return sources_distribution_softmaxed


def invert(sources_distribution):
    sources = [s for s, _ in sources_distribution]
    weights = [1.0 - w for _, w in sources_distribution]

    return list(zip(sources, weights))
