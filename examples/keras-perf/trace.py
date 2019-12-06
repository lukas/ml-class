# https://gist.github.com/tonyreina/5bbe050c2cfceae62a1dda7d9010b692
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np

sess = tf.keras.backend.get_session()
run_options = tf.compat.v1.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.compat.v1.RunMetadata()
model = InceptionV3()

random_input = np.random.random((64, 299, 299, 3))
preds = sess.run(model.output, feed_dict={model.input: random_input},
                 options=run_options, run_metadata=run_metadata)

tl = timeline.Timeline(run_metadata.step_stats)
with open('profile.trace', 'w') as f:
    f.write(tl.generate_chrome_trace_format())

print("Trace saved to profile.trace, open with chrome://tracing")
