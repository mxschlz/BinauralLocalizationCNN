import tensorflow as tf

tf.reset_default_graph()

dtype = tf.float32

# define variables
a = tf.Variable(10, name="a", dtype=dtype)
b = tf.Variable(4, name="b", dtype=dtype)
c = tf.Variable(5, name="c", dtype=dtype)

# save results here
LOGDIR = "/home/max/temp"

# initialize operations
with tf.name_scope("addition"):
    a = tf.Variable(10, name="a", dtype=dtype)
    b = tf.Variable(4, name="b", dtype=dtype)
    five = tf.constant(5, dtype=dtype)
    added = tf.add(tf.add(a, b), five)

with tf.name_scope("exponential"):
    pow = tf.constant(2, dtype=dtype)
    exponential = tf.pow(added, pow)

with tf.name_scope("sqrt"):
    c = tf.Variable(5, name="c", dtype=dtype)
    sqrt = tf.sqrt(c)

with tf.name_scope("final_function"):
    f = tf.add(exponential, sqrt)

# run session
sess = tf.Session()
init = tf.global_variables_initializer()

with sess:
    writer = tf.summary.FileWriter(LOGDIR, sess.graph)
    sess.run(init)
    result = sess.run(f)
    print(result)

