import numpy as np


def reject(t, d):

    return(False)


def fsa_acceptance_probability(t, d):

    return(np.exp(-d / t) > np.random.rand())
