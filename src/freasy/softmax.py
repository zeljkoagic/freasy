import numpy as np


def softmax(sources_distribution, temperature=1.0):
    sources = [s for s, _ in sources_distribution]
    weights = np.array([w for _, w in sources_distribution])

    e = np.exp(weights / temperature)
    softmaxed = e / np.sum(e)

    sources_distribution_softmaxed = list(zip(sources, softmaxed))

    return sources_distribution_softmaxed

#def softmax(sentence_matrix, temperature=1.0):
#    m_exp = np.exp(sentence_matrix/temperature)
#    return (m_exp.T / np.sum(m_exp, axis=1)).T


def invert(sources_distribution):
    sources = [s for s, _ in sources_distribution]
    weights = [1.0 - w for _, w in sources_distribution]

    return list(zip(sources, weights))
