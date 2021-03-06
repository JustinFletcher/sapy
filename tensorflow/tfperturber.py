

import random
import numpy as np
import tensorflow as tf


# TODO: Modularize class to be a single perturber with aniotropicity
# and perturbation distribution decided by string.

class CSATensorFlowPerturber(object):

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
                perturb_state.append(tf.assign_add(v, p,
                                                   name="perturb_state_csa"))

            self.perturb_state = perturb_state

        def start(self, sess):

            self.sess = sess

        def perturb(self, _):

            return(self.sess.run(self.perturb_state))


class FSATensorFlowPerturber(object):

        def __init__(self, learning_rate):

            self.learning_rate = learning_rate

            perturb_state = []

            # Iterate over each prior Variable and trainable variable pair...
            for v in tf.trainable_variables():

                p = tf.tan(np.pi * (tf.random_uniform(v.get_shape()) - 0.5))

                scaled_p = tf.multiply(self.learning_rate, p,
                                       name="make_perturbation")

                # Add an op that perturbs this trainable var
                perturb_state.append(tf.assign_add(v, scaled_p,
                                                   name="perturb_state_fsa"))

            self.perturb_state = perturb_state

        def start(self, sess):

            self.sess = sess

        def perturb(self, _):

            return(self.sess.run(self.perturb_state))


class GSATensorFlowPerturber(object):

        def __init__(self, learning_rate, q):

            self.learning_rate = learning_rate

            perturb_state = []

            # Iterate over each prior Variable and trainable variable pair...
            for v in tf.trainable_variables():

                # GSA perturbation.

                u_1 = tf.random_uniform(v.get_shape())

                u_2 = tf.random_uniform(v.get_shape())

                p_cos = tf.cos(2.0 * np.pi * u_1)

                p_sqrt = tf.sqrt(2 - (2 * (u_2 ** (1 - ((1 + q) / (3 - q))))) / (1 - q))

                p = tf.multiply(p_cos, p_sqrt)

                scaled_p = tf.multiply(self.learning_rate, p,
                                       name="make_perturbation")

                # Add an op that perturbs this trainable var
                perturb_state.append(tf.assign_add(v, scaled_p,
                                                   name="perturb_state_gsa"))

            self.perturb_state = perturb_state

        def start(self, sess):

            self.sess = sess

        def perturb(self, _):

            return(self.sess.run(self.perturb_state))


class LayerwiseCSATensorFlowPerturber(object):

        def __init__(self, learning_rate):

            self.learning_rate = learning_rate

            self.tv_select = tf.placeholder(tf.int32,
                                            shape=[],
                                            name="tv_select")

            perturb_state = []

            # Iterate over each prior Variable and trainable variable pair...
            for tv_num, v in enumerate(tf.trainable_variables()):

                # TODO: Determine if conv ops have 0s we're overwriting

                do = tf.random_normal(v.get_shape())

                do_not = tf.zeros(v.get_shape())

                p = tf.cond(tf.equal(self.tv_select, tv_num),
                            lambda: do,
                            lambda: do_not)

                scaled_p = tf.multiply(self.learning_rate, p,
                                       name="make_perturbation")

                # Add an op that perturbs this trainable var
                perturb_state.append(tf.assign_add(v, scaled_p,
                                                   name="perturb_state_crlsa"))

            self.perturb_state = perturb_state

        def start(self, sess):

            self.sess = sess

        def perturb(self, tv_select):

            tv_count = len(tf.trainable_variables())

            tv_select = random.choice(range(tv_count))

            return(self.sess.run(self.perturb_state,
                                 feed_dict={self.tv_select: tv_select}))


class LayerwiseFSATensorFlowPerturber(object):

        def __init__(self, learning_rate):

            self.learning_rate = learning_rate

            self.tv_select = tf.placeholder(tf.int32,
                                            shape=[],
                                            name="tv_select")

            perturb_state = []

            # Iterate over each prior Variable and trainable variable pair...
            for tv_num, v in enumerate(tf.trainable_variables()):

                # TODO: Determine if conv ops have 0s we're overwriting

                do = tf.tan(np.pi * (tf.random_uniform(v.get_shape()) - 0.5))

                do_not = tf.zeros(v.get_shape())

                p = tf.cond(tf.equal(self.tv_select, tv_num),
                            lambda: do,
                            lambda: do_not)

                scaled_p = tf.multiply(self.learning_rate, p,
                                       name="make_perturbation")

                # Add an op that perturbs this trainable var
                perturb_state.append(tf.assign_add(v, scaled_p,
                                                   name="perturb_state_rlfsa"))

            self.perturb_state = perturb_state

        def start(self, sess):

            self.sess = sess

        def perturb(self, tv_select):

            tv_count = len(tf.trainable_variables())

            tv_select = random.choice(range(tv_count))

            return(self.sess.run(self.perturb_state,
                                 feed_dict={self.tv_select: tv_select}))


class LayerwiseGSATensorFlowPerturber(object):

        def __init__(self, learning_rate, q):

            self.learning_rate = learning_rate

            self.tv_select = tf.placeholder(tf.int32,
                                            shape=[],
                                            name="tv_select")

            perturb_state = []

            # Iterate over each prior Variable and trainable variable pair...
            for tv_num, v in enumerate(tf.trainable_variables()):

                # TODO: Determine if conv ops have 0s we're overwriting

                # GSA perturbation.

                u_1 = tf.random_uniform(v.get_shape())

                u_2 = tf.random_uniform(v.get_shape())

                p_cos = tf.cos(2.0 * np.pi * u_1)

                p_sqrt = tf.sqrt(2 - (2 * (u_2 ** (1 - ((1 + q) / (3 - q))))) / (1 - q))

                p = tf.multiply(p_cos, p_sqrt)

                do = p

                do_not = tf.zeros(v.get_shape())

                p = tf.cond(tf.equal(self.tv_select, tv_num),
                            lambda: do,
                            lambda: do_not)

                scaled_p = tf.multiply(self.learning_rate, p,
                                       name="make_perturbation")
                # Add an op that perturbs this trainable var
                perturb_state.append(tf.assign_add(v, scaled_p,
                                                   name="perturb_state_rlgsa"))

            self.perturb_state = perturb_state

        def start(self, sess):

            self.sess = sess

        def perturb(self, tv_select):

            tv_count = len(tf.trainable_variables())

            tv_select = random.choice(range(tv_count))

            return(self.sess.run(self.perturb_state,
                                 feed_dict={self.tv_select: tv_select}))
