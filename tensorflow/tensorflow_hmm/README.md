[![Build Status](https://travis-ci.org/dwiel/tensorflow_hmm.svg?branch=master)](https://travis-ci.org/dwiel/tensorflow_hmm)

# tensorflow_hmm
Tensorflow and numpy implementations of the HMM viterbi and forward/backward algorithms.

See [Keras example](https://github.com/dwiel/tensorflow_hmm/blob/master/tensorflow_hmm/hmm_layer.py) for an example of how to use the Keras HMMLayer.

See [test_hmm.py](https://github.com/dwiel/tensorflow_hmm/blob/master/test/test_hmm.py) for usage examples.  Here is an excerpt of the documentation from hmm.py for reference for now.

See also viterbi_wikipedia_example.py which replicates the viterbi example on wikipedia.

```
class HMM(object):
    """
    A class for Hidden Markov Models.

    The model attributes are:
    - K :: the number of states
    - P :: the K by K transition matrix (from state i to state j,
        (i, j) in [1..K])
    - p0 :: the initial distribution (defaults to starting in state 0)
    """

    def __init__(self, P, p0=None):

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
      
      
      def viterbi_decode(self, y, nT):
          """
          Runs viterbi decode on state probabilies y.
      
          Arguments
          ---------
          y : np.array : shape (T, K) where T is number of timesteps and
              K is the number of states
          nT : int : number of timesteps in y
      
          Returns
          -------
          (s, pathScores)
          s : list of length T of tensorflow ints : represents the most likely
              state at each time step.
          pathScores : list of length T of tensorflow tensor of length K
              each value at (t, k) is the log likliehood score in state k at
              time t.  sum(pathScores[t, :]) will not necessary == 1
          """
```
