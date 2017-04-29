
from sklearn import datasets
import numpy as np
import tensorflow as tf

sess = tf.InteractiveSession()

digits = datasets.load_digits()

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])  # 5x5 grid
b_conv1 = bias_variable([32])

x = tf.placeholder(tf.float32, shape=[None, 64])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [-1,8,8,1]) # fit data into a tensor

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([2 * 2 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 2*2*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])



y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

dense_target = np.zeros([digits.target.size, 10])
for i in range(digits.target.size):
    dense_target[i, digits.target[i]] = 1

for i in range(1000):
  batch_idxs = np.random.randint(1,1000, 50)

  batch_xs = digits.data[batch_idxs]
  batch_ys = dense_target[batch_idxs]



  if i%10 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch_xs, y_: batch_ys, keep_prob: 1.0})

    print("step %d, training accuracy %g"%(i, train_accuracy))

    test_accuracy = accuracy.eval(feed_dict={
            x:digits.data[1000:1100], y_: dense_target[1000:1100], keep_prob: 1.0})

    print("test accuracy %g"%(test_accuracy))


  train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

#correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

test_accuracy = accuracy.eval(feed_dict={
        x:batch_xs[1000:1100], y_: batch_ys[1000:1100], keep_prob: 1.0})

print("test accuracy %g"%(i, test_accuracy))
