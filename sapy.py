# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 08:20:37 2016

@author: Justin Fletcher
"""
import numpy as np


class Annealer(object):

    def __init__(self,
                 state,
                 state_perturber,
                 state_cost_evaluator,
                 temperature_function,
                 acceptance_function,
                 initial_temperature):

        # Initialize annealing governing functions.
        self.state_perturber = state_perturber
        self.state_cost_evaluator = state_cost_evaluator
        self.state = state
        self.temperature_function = temperature_function
        self.acceptance_function = acceptance_function
        self.initial_temperature = initial_temperature

        # Intialize the annealing state variables.
        self.epoch = 0
        self.temperature = initial_temperature
        self.current_cost = []

    def update_temperature(self):

        self.temperature = self.temperature_function(self.epoch,
                                                     self.initial_temperature)

    def reanneal(self):

        self.epoch = 0
        self.temperature = self.initial_temperature

    def __call__(self, perturb_params=[], input_data=[]):

        if self.epoch == 0:

            self.current_cost = self.state_cost_evaluator.evaluate(input_data)

        # Incement the epoch count.
        self.epoch = self.epoch + 1

        # print("self.epoch = %.d" % self.epoch)

        # Update the temperature.
        self.update_temperature()

        # print("self.temperature = %.6f" % self.temperature)

        # Store the current state.
        self.state.store()

        # Create a new state using the perturbation function.
        self.state_perturber.perturb(perturb_params)

        # Compute the cost of this state using the provided grader.
        candidate_cost = self.state_cost_evaluator.evaluate(input_data)

        # Compute the cost function delta induced by this state.
        cost_delta = candidate_cost - self.current_cost

        # print("cost_delta = %.6f" % cost_delta)

        # If the candidate state reduces the cost...
        if(cost_delta <= 0.0):

            # print("Accept 1")
            # ...accept the candidate state implicitly by updating the cost.
            self.current_cost = candidate_cost

        # If the cost is increased...
        else:

            # print('np.exp(-d / t) = %.9f' % np.exp(-cost_delta / self.temperature))
            # ...and if the acceptance function returns True...
            if(self.acceptance_function(self.temperature, cost_delta)):

                # print("Accept 2")
                # ...accept the state implicitly by updating the cost.
                self.current_cost = candidate_cost

            # If not, we reject the state by restoring the prior state.
            else:

                # print("Reject")
                self.state.restore()

        return()


if __name__ == '__main__':

    my_annealer = Annealer()
