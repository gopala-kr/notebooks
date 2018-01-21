import tensorflow as tf
import numpy as np

x = tf.constant([[1, 2]])
neg_x = tf.neg(x)

print(neg_x)

with tf.Session() as sess:
    result = sess.run(neg_x)
print(result)
