"""
An example using both tensorflow and numpy implementations of viterbi
replicating example on wikipedia
"""
from __future__ import print_function

__author__ = 'Marvin Bertin <marvin.bertin@gmail.com>'

import tensorflow as tf
import numpy as np


from tensorflow_hmm import HMMNumpy, HMMTensorflow


def dptable(V, pathScores, states):
    print(" ".join(("%10d" % i) for i in range(V.shape[0])))
    for i, y in enumerate(pathScores.T):
        print("%.7s: " % states[i])
        print(" ".join("%.7s" % ("%f" % yy) for yy in y))


def main():
    p0 = np.array([0.6, 0.4])

    emi = np.array([[0.5, 0.1],
                    [0.4, 0.3],
                    [0.1, 0.6]])

    trans = np.array([[0.7, 0.3],
                      [0.4, 0.6]])
    states = {0: 'Healthy', 1: 'Fever'}
    obs = {0: 'normal', 1: 'cold', 2: 'dizzy'}

    obs_seq = np.array([0, 1, 2])

    print()
    print("TensorFlow Example: ")

    tf_model = HMMTensorflow(trans, p0)

    y = emi[obs_seq]
    tf_s_graph, tf_scores_graph = tf_model.viterbi_decode(y)
    tf_s = tf.Session().run(tf_s_graph)
    print("Most likely States: ", [obs[s] for s in tf_s])

    tf_scores = tf.Session().run(tf_scores_graph)
    pathScores = np.array(np.exp(tf_scores))
    dptable(pathScores, pathScores, states)

    print()
    print("numpy Example: ")
    np_model = HMMNumpy(trans, p0)

    y = emi[obs_seq]
    np_states, np_scores = np_model.viterbi_decode(y)
    print("Most likely States: ", [obs[s] for s in np_states])
    pathScores = np.array(np.exp(np_scores))
    dptable(pathScores, pathScores, states)

if __name__ == "__main__":
    main()
