import tensorflow as tf
sess = tf.InteractiveSession()

matrix = tf.constant([[1., 2.]])
negMatrix = tf.neg(matrix)

result = negMatrix.eval()
print(result)
sess.close()
