
import numpy as np
import tensorflow as tf


class TensorFlowPerturberGSA(object):

        def __init__(self, learning_rate, q):

            self.learning_rate = learning_rate

            perturb_state = []

            # Iterate over each prior Variable and trainable variable pair...
            for v in tf.trainable_variables():

                # GSA perturbation.
                p = tf.cos(2.0 * np.pi * tf.random_uniform(v.get_shape())) * tf.sqrt(-2.0 * tf.log(tf.random_uniform(v.get_shape())))

                beta = tf.div(1.0, (3.0 - q))

                p_adj = tf.div(p, tf.sqrt(tf.multiply(beta, (3.0 - q))))

                scaled_p = tf.multiply(self.learning_rate, p_adj,
                                       name="make_perturbation")

                # Add an op that perturbs this trainable var
                perturb_state.append(tf.assign_add(v, scaled_p,
                                                   name="perturb_state_gsa"))

            self.perturb_state = perturb_state

        def start(self, sess):

            self.sess = sess

        def perturb(self):

            return(self.sess.run(self.perturb_state))


class TensorFlowPerturberFSA(object):

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

        def perturb(self):

            return(self.sess.run(self.perturb_state))


class TensorFlowPerturberLayerwiseFSA(object):

        def __init__(self, learning_rate):

            self.learning_rate = learning_rate

            tv_select = tf.placeholder(tf.int32, shape=[], name="tv_select")

            perturb_state = []

            # Iterate over each prior Variable and trainable variable pair...
            for tv_num, v in enumerate(tf.trainable_variables()):

                # TODO: Determine if conv ops have 0s we're overwriting

                do = tf.tan(np.pi * (tf.random_uniform(v.get_shape()) - 0.5))

                do_not = tf.zeros(v.get_shape())

                p = tf.cond(tv_select == tv_num, lambda: do, lambda: do_not)

                scaled_p = tf.multiply(self.learning_rate, p,
                                       name="make_perturbation")

                # Add an op that perturbs this trainable var
                perturb_state.append(tf.assign_add(v, scaled_p,
                                                   name="perturb_state_fsa"))

            self.perturb_state = perturb_state

        def start(self, sess):

            self.sess = sess

        def perturb(self, tv_select):

            return(self.sess.run(self.perturb_state,
                                 feed_dict={tv_select: tv_select}))


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
                perturb_state.append(tf.assign_add(v, p,
                                     name="perturb_state_csa"))

            self.perturb_state = perturb_state

        def start(self, sess):

            self.sess = sess

        def perturb(self):

            return(self.sess.run(self.perturb_state))
