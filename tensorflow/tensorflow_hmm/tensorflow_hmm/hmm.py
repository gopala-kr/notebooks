import tensorflow as tf
import numpy as np


class HMM(object):
    """
    A class for Hidden Markov Models.

    The model attributes are:
    - K :: the number of states
    - P :: the K by K transition matrix (from state i to state j,
        (i, j) in [1..K])
    - p0 :: the initial distribution (defaults to starting in state 0)
    """

    def __init__(self, P, p0=None, length=None):
        self.K = P.shape[0]
        self.length = length

        if len(P.shape) !=  2:
            raise ValueError('P shape should have length 2. found {}'.format(len(P.shape)))
        if P.shape[0] !=  P.shape[1]:
            raise ValueError('P.shape should be square, found {}'.format(P.shape))

        # make sure probability matrix is normalized
        P = P / np.sum(P,1)

        self.P = P.astype(dtype=np.float32)
        self.logP = np.log(self.P)

        if p0 is None:
            self.p0 = np.ones(self.K)
            self.p0 /= sum(self.p0)
        elif len(p0) != self.K:
            raise ValueError(
                'dimensions of p0 {} must match P[0] {}'.format(
                    p0.shape, P.shape[0]))
        else:
            self.p0 = p0
        self.logp0 = np.log(self.p0)


class HMMNumpy(HMM):

    def forward_backward(self, y):
        # set up
        if y.ndim == 2:
            y = y[np.newaxis, ...]

        nB, nT = y.shape[:2]

        posterior = np.zeros((nB, nT, self.K))
        forward = np.zeros((nB, nT + 1, self.K))
        backward = np.zeros((nB, nT + 1, self.K))

        # forward pass
        forward[:, 0, :] = 1.0 / self.K
        for t in range(nT):
            tmp = np.multiply(
                np.matmul(forward[:, t, :], self.P),
                y[:, t]
            )
            # normalize
            forward[:, t + 1, :] = tmp / np.sum(tmp, axis=1)[:, np.newaxis]

        # backward pass
        backward[:, -1, :] = 1.0 / self.K
        for t in range(nT, 0, -1):
            # TODO[marcel]: double check whether y[:,t-1] should be y[:,t]
            tmp = np.matmul(self.P, (y[:, t - 1] * backward[:, t, :]).T).T
            # normalize
            backward[:, t - 1, :] = tmp / np.sum(tmp, axis=1)[:, np.newaxis]

        # remove initial/final probabilities and squeeze for non-batched tests
        forward = np.squeeze(forward[:, 1:, :])
        backward = np.squeeze(backward[:, :-1, :])

        # TODO[marcel]: posterior missing initial probabilities
        # combine and normalize
        posterior = np.array(forward) * np.array(backward)
        # [:,None] expands sum to be correct size
        posterior = posterior / np.sum(posterior, axis=-1)[..., np.newaxis]

        # squeeze for non-batched tests
        return posterior, forward, backward

    def _viterbi_partial_forward(self, scores):
        tmpMat = np.zeros((self.K, self.K))
        for i in range(self.K):
            for j in range(self.K):
                tmpMat[i, j] = scores[i] + self.logP[i, j]
        return tmpMat

    def _viterbi_partial_forward_batched(self, scores):
        """
        Expects inputs in [B, K] layout
        """
        # support non-batched version
        if scores.ndim == 1:
            scores = scores[np.newaxis, ...]

        nB, K  = scores.shape
        assert K == self.K, "Incompatible scores"

        tmpMat = np.zeros((nB, self.K, self.K))
        for i in range(self.K):
            for j in range(self.K):
                tmpMat[:, i, j] = scores[:, i] + self.logP[i, j]
        return tmpMat

    def viterbi_decode(self, y):
        nT = y.shape[0]

        pathStates = np.zeros((nT, self.K), dtype=np.int)
        pathScores = np.zeros((nT, self.K))

        # initialize
        pathScores[0] = self.logp0 + np.log(y[0])

        for t, yy in enumerate(y[1:]):
            # propagate forward
            tmpMat = self._viterbi_partial_forward(pathScores[t])
            # the inferred state
            pathStates[t + 1] = np.argmax(tmpMat, 0)
            pathScores[t + 1] = np.max(tmpMat, 0) + np.log(yy)

        # now backtrack viterbi to find states
        s = np.zeros(nT, dtype=np.int)
        s[-1] = np.argmax(pathScores[-1])
        for t in range(nT - 1, 0, -1):
            s[t - 1] = pathStates[t, s[t]]

        return s, pathScores

    def viterbi_decode_batched(self, y):
        """
        Expects inputs in [B, N, K] layout
        """
        # take care of non-batched version
        if y.ndim == 2:
            y = y[np.newaxis, ...]

        nB, nT = y.shape[:2]

        pathStates = np.zeros((nB, nT, self.K), dtype=np.int)
        pathScores = np.zeros((nB, nT, self.K))

        # initialize
        pathScores[:, 0] = self.logp0 + np.log(y[:, 0])

        for t in range(0, nT-1):
            yy = y[:, t+1]
            # propagate forward
            tmpMat = self._viterbi_partial_forward_batched(pathScores[:, t])
            # the inferred state
            pathStates[:, t + 1] = np.argmax(tmpMat, axis=1)
            pathScores[:, t + 1] = np.squeeze(np.max(tmpMat, axis=1)) + np.log(yy)

        # now backtrack viterbi to find states
        s = np.zeros((nB, nT), dtype=np.int)
        s[:, -1] = np.argmax(pathScores[:, -1], axis=1)
        for t in range(nT - 1, 0, -1):
            # s[:, t - 1] = pathStates[:, t][range(nB), s[:, t]]
            s[:, t - 1] = np.choose(s[:, t], pathStates[:, t].T)

        return s, pathScores


def tf_map(fn, arrays):
    """
    Apply fn to each of the values in each of the arrays.  Implemented in
    native python would look like:

        return map(fn, *arrays)

    more explicitly:

        output[i] = fn(arrays[0][i], arrays[1][i], ... arrays[-1][i])

    This function assumes that all arrays have same leading dim.
    """
    indices = tf.range(tf.shape(arrays[0])[0])
    out = tf.map_fn(lambda ii: fn(*[array[ii] for array in arrays]), indices, dtype=tf.int64)
    return out

class HMMTensorflow(HMM):

    def forward_backward(self, y):
        """
        runs forward backward algorithm on state probabilities y

        Arguments
        ---------
        y : np.array : shape (T, K) where T is number of timesteps and
            K is the number of states

        Returns
        -------
        (posterior, forward, backward)
        posterior : list of length T of tensorflow graph nodes representing
            the posterior probability of each state at each time step
        forward : list of length T of tensorflow graph nodes representing
            the forward probability of each state at each time step
        backward : list of length T of tensorflow graph nodes representing
            the backward probability of each state at each time step
        """
        y = tf.cast(y, tf.float32)

        if len(y.shape) == 2:
            y = tf.expand_dims(y, axis=0)

        # set up
        N = tf.shape(y)[0]

        # y (batch, recurrent, features) -> (recurrent, batch, features)
        y = tf.transpose(y, (1, 0, 2))

        # forward pass
        def forward_function(last_forward, yi):
            tmp = tf.multiply(tf.matmul(last_forward, self.P), yi)
            return tmp / tf.reduce_sum(tmp, axis=1, keep_dims=True)

        forward = tf.scan(
            forward_function,
            y,
            initializer=tf.ones((N, self.K)) * (1.0 / self.K),
        )

        # backward pass
        def backward_function(last_backward, yi):
            # combine transition matrix with observations
            combined = tf.multiply(
                tf.expand_dims(self.P, 0), tf.expand_dims(yi, 1)
            )
            tmp = tf.reduce_sum(
                tf.multiply(combined, tf.expand_dims(last_backward, 1)), axis=2
            )
            return tmp / tf.reduce_sum(tmp, axis=1, keep_dims=True)

        backward = tf.scan(
            backward_function,
            tf.reverse(y, [0]),
            initializer=tf.ones((N, self.K)) * (1.0 / self.K),
        )
        backward = tf.reverse(backward, [0])

		# combine forward and backward into posterior probabilities
        # (recurrent, batch, features)
        posterior = forward * backward
        posterior = posterior / tf.reduce_sum(posterior, axis=2, keep_dims=True)

        # (recurrent, batch, features) -> (batch, recurrent, features)
        posterior = tf.transpose(posterior, (1, 0, 2))
        forward = tf.transpose(forward, (1, 0, 2))
        backward = tf.transpose(backward, (1, 0, 2))

        return posterior, forward, backward

    def _viterbi_partial_forward(self, scores):
        # first convert scores into shape [K, 1]
        # then concatenate K of them into shape [K, K]
        expanded_scores = tf.concat(
            [tf.expand_dims(scores, 1)] * self.K, 1
        )
        return expanded_scores + self.logP

    def _viterbi_partial_forward_batched(self, scores):
        """
        Expects inputs in [B, Kl layout
        """
        # first convert scores into shape [B, K, 1]
        # then concatenate K of them into shape [B, K, K]
        expanded_scores = tf.concat(
            [tf.expand_dims(scores, axis=2)] * self.K, axis=2
        )
        return expanded_scores + self.logP

    def viterbi_decode(self, y):
        """
        Runs viterbi decode on state probabilies y.

        Arguments
        ---------
        y : np.array : shape (T, K) where T is number of timesteps and
            K is the number of states

        Returns
        -------
        (s, pathScores)
        s : list of length T of tensorflow ints : represents the most likely
            state at each time step.
        pathScores : list of length T of tensorflow tensor of length K
            each value at (t, k) is the log likliehood score in state k at
            time t.  sum(pathScores[t, :]) will not necessary == 1
        """
        y = np.asarray(y)
        if len(y.shape) != 2:
            raise ValueError((
                'y should be 2d of shape (nT, {}).  Found {}'
            ).format(self.K, y.shape))

        if y.shape[1] != self.K:
            raise ValueError((
                'y has an invalid shape.  first dimension is time and second '
                'is K.  Expected K for this model is {}, found {}.'
            ).format(self.K, y.shape[1]))

        nT = y.shape[0]

        # pathStates and pathScores wil be of type tf.Tensor.  They
        # are lists since tensorflow doesn't allow indexing, and the
        # list and order are only really necessary to build the unrolled
        # graph.  We never do any computation across all of time at once
        pathStates = []
        pathScores = []

        # initialize
        pathStates.append(None)
        pathScores.append(self.logp0 + np.log(y[0]))

        for t, yy in enumerate(y[1:]):
            # propagate forward
            tmpMat = self._viterbi_partial_forward(pathScores[t])

            # the inferred state
            pathStates.append(tf.argmax(tmpMat, 0))
            pathScores.append(tf.reduce_max(tmpMat, 0) + np.log(yy))

        # now backtrack viterbi to find states
        s = [0] * nT
        s[-1] = tf.argmax(pathScores[-1], 0)
        for t in range(nT - 1, 0, -1):
            s[t - 1] = tf.gather(pathStates[t], s[t])

        return s, tf.stack(pathScores, axis=0)

    def viterbi_decode_batched(self, y, onehot=False):
        """
        Runs viterbi decode on state probabilies y in batch mode

        Arguments
        ---------
        y : np.array : shape (B, T, K) where T is number of timesteps and
            K is the number of states
        onehot : boolean : if true, returns a onehot representation of the
            most likely states, instead of integer indexes of the most likely
            states.

        Returns
        -------
        (s, pathScores)
        s : list of length T of tensorflow ints : represents the most likely
            state at each time step.
        pathScores : list of length T of tensorflow tensor of length K
            each value at (t, k) is the log likliehood score in state k at
            time t.  sum(pathScores[t, :]) will not necessary == 1
        """
        if len(y.shape) == 2:
            # y = y[np.newaxis, ...]
            y = tf.expand_dims(y, axis=0)

        if  len(y.shape) != 3:
            raise ValueError((
                'y should be 3d of shape (nB, nT, {}).  Found {}'
            ).format(self.K, y.shape))

        nB, nT, nC = y.shape

        if nC != self.K:
            raise ValueError((
                'y has an invalid shape.  first dimension is time and second '
                'is K.  Expected K for this model is {}, found {}.'
            ).format(self.K, nC))

        # pathStates and pathScores will be of type tf.Tensor.  They
        # are lists since tensorflow doesn't allow indexing, and the
        # list and order are only really necessary to build the unrolled
        # graph.  We never do any computation across all of time at once. The
        # indexing into these list has the dimension of time.
        pathStates = []
        pathScores = []

        # initialize
        pathStates.append(None)
        pathScores.append(self.logp0 + tf.log(y[:, 0]))

        for t in range(0, nT-1):
            yy = tf.squeeze(y[:, t+1])
            # propagate forward
            tmpMat = self._viterbi_partial_forward_batched(pathScores[t])

            # the inferred state
            pathStates.append(tf.argmax(tmpMat, axis=1))
            pathScores.append(tf.reduce_max(tmpMat, axis=1) + tf.log(yy))

        # now backtrack viterbi to find states
        s = [None] * nT
        s[-1] = tf.argmax(pathScores[-1], axis=1)
        for t in range(nT - 1, 0, -1):
            s[t - 1] = tf_map(lambda p, i: p[i], [pathStates[t], s[t]])

        s = tf.stack(s, axis=1)
        pathScores = tf.stack(pathScores, axis=1)

        if onehot:
            s = tf.one_hot(s, depth=nC)

        return s, pathScores
