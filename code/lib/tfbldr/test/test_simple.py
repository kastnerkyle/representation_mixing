import logging
logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
import numpy as np
from tfbldr import make_numpy_weights, make_numpy_biases, dot, scan, get_params_dict
from tfbldr.nodes import Linear, SimpleRNNCell

n_batch = 64
h_dim = 400
random_state = np.random.RandomState(2145)

inputs = tf.placeholder(tf.float32, [None, n_batch, 3],
                            name="inputs")
init_h = tf.placeholder(tf.float32, [n_batch, h_dim],
                            name="init_h")

def step(inp_t, h_tm1):
    output, state = SimpleRNNCell([inp_t], [3], h_tm1, h_dim, 20, random_state=random_state,
                              name="l1")
    h = state[0]
    return output, h

o = scan(step, [inputs], [None, init_h])
loss = tf.reduce_mean(o[0])
h_o = o[1]

params_dict = get_params_dict()
params = params_dict.values()
grads = tf.gradients(loss, params)

learning_rate = 0.0002
opt = tf.train.AdamOptimizer(learning_rate=learning_rate, use_locking=True)
updates = opt.apply_gradients(zip(grads, params))

inputs_np = random_state.randn(33, n_batch, 3)
init_h_np = np.zeros((n_batch, h_dim))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    feed = {inputs: inputs_np,
            init_h: init_h_np}
    outs = [loss, updates, h_o]
    lop = sess.run(outs, feed)
