import tensorflow as tf
x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2

with tf.Session() as sess:
        x.initializer.run()
        y.initializer.run()
        result = f.eval()
        print(result)
