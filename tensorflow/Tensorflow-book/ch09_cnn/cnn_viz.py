import numpy as np
import matplotlib.pyplot as plt
import cifar_tools
import tensorflow as tf

learning_rate = 0.001

names, data, labels = \
    cifar_tools.read_data('/home/binroot/res/cifar-10-batches-py')

x = tf.placeholder(tf.float32, [None, 24 * 24], name='input')
y = tf.placeholder(tf.float32, [None, len(names)], name='prediction')
W1 = tf.Variable(tf.random_normal([5, 5, 1, 64]), name='W1')
b1 = tf.Variable(tf.random_normal([64]), name='b1')
W2 = tf.Variable(tf.random_normal([5, 5, 64, 64]), name='W2')
b2 = tf.Variable(tf.random_normal([64]), name='b2')
W3 = tf.Variable(tf.random_normal([6*6*64, 1024]), name='W3')
b3 = tf.Variable(tf.random_normal([1024]), name='b3')
W_out = tf.Variable(tf.random_normal([1024, len(names)]), name='W_out')
b_out = tf.Variable(tf.random_normal([len(names)]), name='b_out')


W1_summary = tf.image_summary('W1_img', W1)

def conv_layer(x, W, b):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    conv_with_b = tf.nn.bias_add(conv, b)
    conv_out = tf.nn.relu(conv_with_b)
    return conv_out


def maxpool_layer(conv, k=2):
    return tf.nn.max_pool(conv, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def model():
    x_reshaped = tf.reshape(x, shape=[-1, 24, 24, 1])

    conv_out1 = conv_layer(x_reshaped, W1, b1)
    maxpool_out1 = maxpool_layer(conv_out1)
    norm1 = tf.nn.lrn(maxpool_out1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    conv_out2 = conv_layer(norm1, W2, b2)
    norm2 = tf.nn.lrn(conv_out2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    maxpool_out2 = maxpool_layer(norm2)

    maxpool_reshaped = tf.reshape(maxpool_out2, [-1, W3.get_shape().as_list()[0]])
    local = tf.add(tf.matmul(maxpool_reshaped, W3), b3)
    local_out = tf.nn.relu(local)

    out = tf.add(tf.matmul(local_out, W_out), b_out)
    return out

model_op = model()

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model_op, y))
tf.scalar_summary('cost', cost)
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(model_op, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

merged = tf.merge_all_summaries()

with tf.Session() as sess:
    summary_writer = tf.train.SummaryWriter('summaries/train', sess.graph)
    sess.run(tf.initialize_all_variables())
    onehot_labels = tf.one_hot(labels, len(names), on_value=1., off_value=0., axis=-1)
    onehot_vals = sess.run(onehot_labels)
    batch_size = len(data) / 200
    print('batch size', batch_size)
    for j in range(0, 1000):
        print('EPOCH', j)
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i+batch_size, :]
            batch_onehot_vals = onehot_vals[i:i+batch_size, :]
            _, accuracy_val, summary = sess.run([train_op, accuracy, merged], feed_dict={x: batch_data, y: batch_onehot_vals})
            summary_writer.add_summary(summary, i)
            if i % 1000 == 0:
                print(i, accuracy_val)
        print('DONE WITH EPOCH')































