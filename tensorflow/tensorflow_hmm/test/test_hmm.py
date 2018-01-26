from __future__ import print_function

import pytest
import numpy as np
import tensorflow as tf

from tensorflow_hmm import HMMNumpy, HMMTensorflow


@pytest.fixture
def latch_P():
    P = np.array([[0.5, 0.5], [0.0, 1.0]])
    # P = np.array([[0.5, 0.5], [0.5, 0.5]])
    # P = np.array([[0.5, 0.5], [0.0000000001, 0.9999999999]])
    # P = np.array([[0.5, 0.5], [1e-50, 1 - 1e-50]])

    for i in range(2):
        for j in range(2):
            print('from', i, 'to', j, P[i, j])
    return P


@pytest.fixture
def hmm_latch(latch_P):
    return HMMNumpy(latch_P)


@pytest.fixture
def fair_P():
    return np.array([[0.5, 0.5], [0.5, 0.5]])


@pytest.fixture
def hmm_fair(fair_P):
    return HMMNumpy(fair_P)


@pytest.fixture
def hmm_tf_fair(fair_P):
    return HMMTensorflow(fair_P)


@pytest.fixture
def hmm_tf_latch(latch_P):
    return HMMTensorflow(latch_P)


def lik(y):
    """

    given 1d vector of likliehoods length N, return matrix with
    shape (N, 2) where (N, 0) is 1 - y and (N, 1) is y.

    given a 2d array of likelihood sequences of size [N, B] where B is the batch
    size, return [B, N, 2] where out[B, N, 0] + out[B, N, 1] = 1

    This makes it easy to convert a time series of probabilities
    into 2 states, off/on, for a simple HMM.
    """

    liklihood = np.array([y, y], float).T
    liklihood[..., 0] = 1 - liklihood[..., 0]
    return liklihood

def test_tf_hmm_invalid_P_shape():
    with pytest.raises(ValueError):
        HMMTensorflow(np.ones((1, 2)))

def test_tf_hmm_invalid_P_dimensions():
    with pytest.raises(ValueError):
        HMMTensorflow(np.ones((1,)))

def test_hmm_tf_fair_forward_backward(hmm_tf_fair, hmm_fair):
    y = lik(np.array([0, 0, 1, 1]))

    np_posterior, _, _ = hmm_fair.forward_backward(y)
    print('tf')
    g_posterior, _, _ = hmm_tf_fair.forward_backward(y)
    tf_posterior = np.concatenate(tf.Session().run(g_posterior))

    print('np_posterior', np_posterior)
    print('tf_posterior', tf_posterior)
    assert np.isclose(np_posterior, tf_posterior).all()


def test_hmm_tf_fair_forward_backward_multiple_batch(hmm_tf_fair, hmm_fair):
    y = lik(np.array([0, 0, 1, 1]))
    y = np.stack([y] * 3)

    np_posterior, _, _ = hmm_fair.forward_backward(y)
    print('tf')
    g_posterior, _, _ = hmm_tf_fair.forward_backward(y)
    tf_posterior = tf.Session().run(g_posterior)

    print('np_posterior', np_posterior)
    print('tf_posterior', tf_posterior)
    assert np.isclose(np_posterior, tf_posterior).all()


def test_hmm_tf_latch_forward_backward_multiple_batch(hmm_tf_latch, hmm_latch):
    y = lik(np.array([0, 0, 1, 1]))
    y = np.stack([y] * 3)

    np_posterior, np_forward, np_backward = hmm_latch.forward_backward(y)
    print('tf')
    g_posterior, g_forward, g_backward = hmm_tf_latch.forward_backward(y)
    tf_posterior = tf.Session().run(g_posterior)
    tf_forward = tf.Session().run(g_forward)
    tf_backward = tf.Session().run(g_backward)

    assert np.isclose(np_forward, tf_forward).all()
    print('np_backward', np_backward)
    print('tf_backward', tf_backward)
    assert np.isclose(np_backward, tf_backward).all()
    print('np_posterior', np_posterior)
    print('tf_posterior', tf_posterior)
    assert np.isclose(np_posterior, tf_posterior).all()

def test_lik():
    yin = np.array([0, 0.25, 0.5, 0.75, 1])
    y = lik(yin)

    assert np.all(y == np.array([
        [1.00, 0.00],
        [0.75, 0.25],
        [0.50, 0.50],
        [0.25, 0.75],
        [0.00, 1.00],
    ]))


def test_hmm_fair_forward_backward(hmm_fair):
    y = lik(np.array([0, 0, 1, 1]))

    posterior, f, b = hmm_fair.forward_backward(y)

    # if P is filled with 0.5, the only thing that matters is the emission
    # liklihood.  assert that the posterior is = the liklihood of y
    for i, yi in enumerate(y):
        liklihood = yi / np.sum(yi)
        assert np.isclose(posterior[i, :], liklihood).all()

    # assert that posterior for any given t sums to 1
    assert np.isclose(np.sum(posterior, 1), 1).all()


def test_hmm_latch_two_step_no_noise(hmm_latch):
    for i in range(2):
        for j in range(2):
            y = [i, i, j, j]
            # y = [i, j]

            if i == 1 and j == 0:
                continue

            print('*'*80)
            print(y)
            states, scores = hmm_latch.viterbi_decode(lik(y))

            assert all(states == y)


def test_hmm_tf_partial_forward(hmm_tf_latch, hmm_latch):
    scoress = [
        np.log(np.array([0, 1])),
        np.log(np.array([1, 0])),
        np.log(np.array([0.25, 0.75])),
        np.log(np.array([0.5, 0.5])),
    ]

    for scores in scoress:
        tf_ret = tf.Session().run(
            hmm_tf_latch._viterbi_partial_forward(scores)
        )
        np_ret = hmm_latch._viterbi_partial_forward(scores)

        assert (tf_ret == np_ret).all()


def test_hmm_tf_partial_forward_batched(hmm_tf_latch, hmm_latch):
    scoress = [
        np.log(np.array([0, 1])),
        np.log(np.array([1, 0])),
        np.log(np.array([0.25, 0.75])),
        np.log(np.array([0.5, 0.5])),
    ]

    scores_batch = np.asarray(scoress)

    np_res = hmm_latch._viterbi_partial_forward_batched(scores_batch)
    tf_res = tf.Session().run(
        hmm_tf_latch._viterbi_partial_forward_batched(scores_batch)
    )

    assert (tf_res == np_res).all()


def test_hmm_partial_forward_batched(hmm_latch):
    scoress = [
        np.log(np.array([0, 1])),
        np.log(np.array([1, 0])),
        np.log(np.array([0.25, 0.75])),
        np.log(np.array([0.5, 0.5])),
    ]

    scores_batch = np.array(scoress)

    res = [hmm_latch._viterbi_partial_forward(scores) for scores in scoress]
    res_batched = hmm_latch._viterbi_partial_forward_batched(scores_batch)

    assert np.all(np.asarray(res) == res_batched)


def test_hmm_tf_viterbi_decode(hmm_tf_latch, hmm_latch):
    ys = [
        lik(np.array([0, 0])),
        lik(np.array([1, 1])),
        lik(np.array([0, 1])),
        lik(np.array([0, 0.25, 0.5, 0.75, 1])),
    ]

    for y in ys:
        tf_s_graph, tf_scores_graph = hmm_tf_latch.viterbi_decode(y)
        tf_s, tf_scores = tf.Session().run([tf_s_graph, tf_scores_graph])

        np_s, np_scores = hmm_latch.viterbi_decode(y)

        assert (tf_s == np_s).all()
        assert (tf_scores == np_scores).all()


def test_hmm_viterbi_decode_batched(hmm_latch):
    ys_T2 = [
        lik(np.array([0, 0])),
        lik(np.array([0, 1])),
        lik(np.array([1, 1])),
    ]
    ys_T5 = [
        lik([0, 0.25, 0.5, 0.75, 1]),
        lik([0, 0.65, 0.5, 0.95, .1]),
    ]

    ys_T2_batch = np.asarray(ys_T2)
    ys_T5_batch = np.asarray(ys_T5)

    res = [hmm_latch.viterbi_decode(y) for y in ys_T2]
    res_s, res_scores = zip(*res)
    res_s_batch, res_scores_batch = hmm_latch.viterbi_decode_batched(ys_T2_batch)
    assert np.all(np.asarray(res_s) == res_s_batch)
    assert np.all(np.asarray(res_scores) == res_scores_batch)

    res = [hmm_latch.viterbi_decode(y) for y in ys_T5]
    res_s, res_scores = zip(*res)
    res_s_batch, res_scores_batch = hmm_latch.viterbi_decode_batched(ys_T5_batch)
    assert np.all(np.asarray(res_s) == res_s_batch)
    assert np.all(np.asarray(res_scores) == res_scores_batch)


def test_hmm_tf_viterbi_decode_batched(hmm_tf_latch, hmm_latch):
    ys_T2_batch = np.asarray([
        lik(np.array([0, 0])),
        lik(np.array([0, 1])),
        lik(np.array([1, 1])),
    ], dtype=np.float32)

    ys_T5_batch = np.asarray([
        lik([0, 0.25, 0.5, 0.75, 1]),
        lik([0, 0.65, 0.5, 0.95, .1]),
        lik([0, 0.25, 0.5, 0.75, 1]),
    ], dtype=np.float32)

    for y in (ys_T5_batch, ys_T2_batch):
        np_res_s, np_res_scores = hmm_latch.viterbi_decode_batched(y)

        y_variable = tf.placeholder(tf.float32, shape=(None, y.shape[1], y.shape[2]))
        tf_s_graph, tf_scores_graph = hmm_tf_latch.viterbi_decode_batched(y_variable)
        init_op = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init_op)

            tf_s = session.run(tf_s_graph, {y_variable: y})
            tf_scores = session.run(tf_scores_graph, {y_variable: y})

        np.testing.assert_allclose(tf_s, np_res_s)
        np.testing.assert_allclose(tf_scores, np_res_scores)


def test_hmm_tf_viterbi_decode_wrong_shape(hmm_tf_latch, hmm_latch):
    with pytest.raises(ValueError):
        hmm_tf_latch.viterbi_decode([0, 1, 1, 0])
