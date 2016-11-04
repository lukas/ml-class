from sklearn import datasets
import numpy as np
import tensorflow as tf

digits = datasets.load_digits()


x = tf.placeholder(tf.float32, [None, 64])

W = tf.Variable(tf.zeros([64, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)


dense_target = np.zeros([digits.target.size, 10])
for i in range(digits.target.size):
    dense_target[i, digits.target[i]] = 1

for step in range(10):
    batch_xs = digits.data
    batch_ys = dense_target
    sess_output = sess.run([train_step, loss, W], feed_dict={x: batch_xs, y_: batch_ys})
    print 'Step %d: loss = %.2f' % (step, sess_output[1])



correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: digits.data[1:100], y_: dense_target[1:100]}))
