import tensorflow as tf


class TensorFlowState(object):

    def __init__(self):

        # Get the graph.
        graph = tf.get_default_graph()

        # Extract the global varibles from the graph.
        self.gvars = graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        # Exract the Assign operations for later use.
        self.assign_ops = [graph.get_operation_by_name(v.op.name + "/Assign")
                           for v in self.gvars]

        # Extract the initial value ops from each Assign op for later use.
        self.init_values = [op.inputs[1] for op in self.assign_ops]

    def start(self, sess):

        self.sess = sess

    # State set function.
    def store(self):

        # Record the current state of the TF global varaibles
        self.state = self.sess.run(self.gvars)

    # State set function.
    def restore(self):

        # Create a dictionary of the iniailizers and stored state of globals.
        feed_dict = {init_value: val
                     for init_value, val in zip(self.init_values, self.state)}

        # Use the initializer ops for each variable to load the stored values.
        return(self.sess.run(self.assign_ops, feed_dict=feed_dict))
