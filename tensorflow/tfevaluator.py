

class TensorFlowCostEvaluator(object):

    def __init__(self, loss_op):

        self.loss_op = loss_op

    def start(self, sess):

        self.sess = sess

    def evaluate(self, data):

        return(self.sess.run(self.loss_op, feed_dict=data))


class TensorFlowQueueCostEvaluator(object):

    def __init__(self, loss_op):

        self.loss_op = loss_op

    def start(self, sess):

        self.sess = sess

    def evaluate(self, _):

        return(self.sess.run(self.loss_op))
