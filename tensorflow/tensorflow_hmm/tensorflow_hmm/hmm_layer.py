from keras.layers import Lambda, Activation
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf

from tensorflow_hmm import HMMTensorflow


class HMMLayer(Layer):
    def __init__(self, states, length=None, viterbi_inference=True, **kwargs):
        # todo: perhaps states should just be inferred by the input shape
        # todo: create a few utility functions for generating transition matrices
        self.viterbi_inference = viterbi_inference
        self.states = states
        self.P = np.ones((states, states), dtype=np.float32) * (0.01 / (states - 1))
        for i in range(states):
            self.P[i, i] = 0.99

        self.hmm = HMMTensorflow(self.P)

        super(HMMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError('input_shape must be 3, found {}'.format(len(input_shape)))

        super(HMMLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        # todo: only optionally apply sigmoid
        # todo: apply viterbi during inference
        x = Activation(K.sigmoid)(x)

        # using K.in_train_phase results in both if and else conditions being
        # computed, which in this case is very expensive. instead, tf.cond
        # is used. Even so, if and else conditions must be wrapped in a lambda
        # to ensure that they are not computed unless that path is chosen.
        if self.viterbi_inference:
            # include this in the graph so that keras knows that the learning phase
            # variable needs to be passed into tensorflows session run.
            x = K.in_train_phase(x, x)

            return Lambda(lambda x: tf.cond(
                K.learning_phase(),
                lambda: self.hmm.forward_backward(x)[0],
                lambda: self.hmm.viterbi_decode_batched(x, onehot=True)[0],
            ))(x)
        else:
            return Lambda(lambda x: self.hmm.forward_backward(x)[0])(x)

    def compute_output_shape(self, input_shape):
        return input_shape
