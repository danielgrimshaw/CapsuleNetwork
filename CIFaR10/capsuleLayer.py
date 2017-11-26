import numpy as np
import tensorflow as tf

from config import cfg


epsilon = 1e-9


class CapsLayer(object):
    def __init__(self, num_outputs, vec_len, with_routing=True, layer_type='FC'):
        self.num_outputs = num_outputs
        self.vec_len = vec_len
        self.with_routing = with_routing
        self.layer_type = layer_type

    def __call__(self, input, kernel_size=None, stride=None):
        if self.layer_type == 'CONV':
            self.kernel_size = kernel_size
            self.stride = stride
            
            if not self.with_routing:
                # input: [batch_size, 20, 20, 256]
                assert input.get_shape() == [cfg.batch_size, 20, 20, 256], input.get_shape()
                capsules = tf.contrib.layers.conv2d(input, self.num_outputs * self.vec_len,
                        self.kernel_size, self.stride, padding="VALID",
                        activation_fn=tf.nn.relu)
                capsules = tf.reshape(capsules, (cfg.batch_size, -1, self.vec_len, 1))
                # [batch_size, 1152, 8, 1]
                capsules = squash(capsules)
                assert capsules.get_shape() == [cfg.batch_size, 1152, 8, 1], capsules.get_shape()
                return(capsules)
            
        if self.layer_type == 'FC':
            if self.with_routing:
                # Reshape the input into [batch_size, 1152, 1, 8, 1]
                self.input = tf.reshape(input, shape=(cfg.batch_size, -1, 1, input.shape[-2].value, 1))
                with tf.variable_scope('routing'):
                    # b_IJ: [1, num_caps_l, num_caps_l_plus_1, 1, 1]
                    b_IJ = tf.constant(np.zeros([cfg.batch_size, input.shape[1].value, self.num_outputs, 1, 1], dtype=np.float32))
                    capsules = routing(self.input, b_IJ)
                    capsules = tf.squeeze(capsules, axis=1)
                    return(capsules)

def routing(input, b_IJ):
    # W: [num_caps_j, num_caps_i, len_u_i, len_v_j]
    W = tf.get_variable('Weight', shape=(1, 1152, 10, 8, 16), dtype=tf.float32,
            initializer=tf.random_normal_initializer(stddev=cfg.stddev))
    input = tf.tile(input, [1, 1, 10, 1, 1])
    W = tf.tile(W, [cfg.batch_size, 1, 1, 1, 1])
    assert input.get_shape() == [cfg.batch_size, 1152, 10, 8, 1], input.get_shape()
    u_hat = tf.matmul(W, input, transpose_a=True)
    assert u_hat.get_shape() == [cfg.batch_size, 1152, 10, 16, 1], u_hat.get_shape()

    for r_iter in range(cfg.iter_routing):
        with tf.variable_scope('iter_' + str(r_iter)):
            c_IJ = tf.nn.softmax(b_IJ, dim=2)
            assert c_IJ.get_shape() == [cfg.batch_size, 1152, 10, 1, 1], c_IJ.get_shape()
            s_J = tf.multiply(c_IJ, u_hat)
            s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
            assert s_J.get_shape() == [cfg.batch_size, 1, 10, 16, 1], s_J.get_shape()
            v_J = squash(s_J)
            assert v_J.get_shape() == [cfg.batch_size, 1, 10, 16, 1], v_J.get_shape()
            v_J_tiled = tf.tile(v_J, [1, 1152, 1, 1, 1])
            u_produce_v = tf.matmul(u_hat, v_J_tiled, transpose_a=True)
            assert u_produce_v.get_shape() == [cfg.batch_size, 1152, 10, 1, 1], u_produce_v.get_shape()
            if r_iter < cfg.iter_routing - 1:
                b_IJ += u_produce_v
    return(v_J)

def squash(vector):
    vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keep_dims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * vector  # element-wise
    return(vec_squashed)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
