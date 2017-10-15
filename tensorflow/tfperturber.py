
import numpy as np
import tensorflow as tf


class TensorFlowPerturberFSA(object):

        def __init__(self, learning_rate):

            self.learning_rate = learning_rate

            perturb_state = []

            # Iterate over each prior Variable and trainable variable pair...
            for v in tf.trainable_variables():

                # FSA perturbation.
                p = tf.multiply(self.learning_rate,
                                np.pi *
                                (tf.random_uniform(v.get_shape()) - 0.5),
                                name="make_perturbation")

                # Add an op that perturbs this trainable var
                perturb_state.append(tf.assign_add(v, p, name="perturb_state"))

            self.perturb_state = perturb_state

        def start(self, sess):

            self.sess = sess

        def perturb(self):

            return(self.sess.run(self.perturb_state))


class TensorFlowPerturberCSA(object):

        def __init__(self, learning_rate):

            self.learning_rate = learning_rate

            perturb_state = []

            # Iterate over each prior Variable and trainable variable pair...
            for v in tf.trainable_variables():

                # CSA perturbation.
                p = tf.multiply(self.learning_rate,
                                tf.random_normal(v.get_shape()),
                                name="make_perturbation")

                # Add an op that perturbs this trainable var
                perturb_state.append(tf.assign_add(v, p, name="perturb_state"))

            self.perturb_state = perturb_state

        def start(self, sess):

            self.sess = sess

        def perturb(self):

            return(self.sess.run(self.perturb_state))
