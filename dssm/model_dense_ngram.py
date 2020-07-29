import tensorflow as tf
from dssm.config import NO_OF_TRIGRAMS

NO_OF_INDICES = NO_OF_TRIGRAMS

tf.compat.v1.disable_eager_execution()

# tf Graph Input
x = tf.compat.v1.placeholder("float32", name="x_q", shape=[None, NO_OF_INDICES])

W_2 = tf.compat.v1.Variable(tf.compat.v1.truncated_normal([NO_OF_INDICES, 300], mean=0, stddev=0.1, dtype=tf.float32), name="W2")
b_2 = tf.compat.v1.Variable(tf.compat.v1.truncated_normal([1, 300], mean=0, stddev=0.1, dtype=tf.float32), name="b2")

l2 = tf.compat.v1.tanh(tf.compat.v1.matmul(x, W_2) + b_2)

W_3 = tf.compat.v1.Variable(tf.compat.v1.truncated_normal([300, 300], mean=0, stddev=0.1, dtype=tf.float32), name="W3")
b_3 = tf.compat.v1.Variable(tf.compat.v1.truncated_normal([1, 300], mean=0, stddev=0.1, dtype=tf.float32), name="b3")

l3 = tf.compat.v1.tanh(tf.compat.v1.matmul(l2, W_3) + b_3)

W_4 = tf.compat.v1.Variable(tf.compat.v1.truncated_normal([300, 128], mean=0, stddev=0.1, dtype=tf.float32), name="W4")
b_4 = tf.compat.v1.Variable(tf.compat.v1.truncated_normal([1, 128], mean=0, stddev=0.1, dtype=tf.float32), name="b4")

y = tf.compat.v1.tanh(tf.compat.v1.matmul(l3, W_4) + b_4)


