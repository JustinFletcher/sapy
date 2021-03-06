import numpy as np


# CSA temperature function.
def csa_temperature(i, init_temp):

        return(init_temp / np.log(float(i)))


# FSA temperature function.
def fsa_temperature(i, init_temp):

        return(init_temp / (float(i)))
