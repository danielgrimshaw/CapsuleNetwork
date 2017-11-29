import tensorflow as tf

from config import cfg
from utils import get_batch_data
from capsuleLayer import CapsLayer

epsilon = 1e-9

class CapsNet(object):
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                self.X, self.labels = get_batch_data()
                self.Y = tf.one_hot(self.labels, depth=10, axis=1, dtype=tf.float32)

                self.build_arch()
                self.loss()
                self._summary()

                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer()
                self.train_op = self.optimizer.minimize(self.total_loss, global_step=self.global_step)
            elif cfg.mask_with_y:
                self.X = tf.placeholder(tf.float32,
                        shape=(cfg.batch_size, 28, 28, 1))
                self.Y = tf.placeholder(tf.float32, shape=(cfg.batch_size, 10, 1))
                self.build_arch()
            else:
                self.X = tf.placeholder(tf.float32,
                        shape=(cfg.batch_size, 28, 28, 1))
                self.build_arch()
        tf.logging.info("Setting up the main structure")

    def build_arch(self):
        with tf.variable_scope('Conv1_layer'):
            conv1 = tf.contrib.layers.conv2d(self.X, num_outputs=128,
                    kernel_size=5, stride=1, padding="VALID")
            conv2 = tf.contrib.layers.conv2d(conv1, num_outputs=256,
                    kernel_size=5, stride=1, padding="VALID")
            assert conv2.get_shape() == [cfg.batch_size, 20, 20, 256], conv1.get_shape()

        # Primary Capsule Layer
        with tf.variable_scope('PrimaryCaps_layer'):
            primaryCaps = CapsLayer(num_outputs=32, vec_len=8, with_routing=False, layer_type="CONV")
            caps1 = primaryCaps(conv2, kernel_size=9, stride=2)
            assert caps1.get_shape() == [cfg.batch_size, 1152, 8, 1]

        # Digit Capsule Layer
        with tf.variable_scope('DigitCaps_layer'):
            digitCaps = CapsLayer(num_outputs=10, vec_len=16, with_routing=True, layer_type="FC")
            self.caps2 = digitCaps(caps1)

        # Decoder
        with tf.variable_scope('Masking'):
            self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2),
                axis=2, keep_dims=True) + epsilon)
            self.softmax_v = tf.nn.softmax(self.v_length, dim=1)
            assert self.softmax_v.get_shape() == [cfg.batch_size, 10, 1, 1]

            self.argmax_idx = tf.to_int32(tf.argmax(self.softmax_v, axis=1))
            assert self.argmax_idx.get_shape() == [cfg.batch_size, 1, 1]
            self.argmax_idx = tf.reshape(self.argmax_idx, shape=(cfg.batch_size, ))

            if not cfg.mask_with_y:
                masked_v = []
                for batch_size in range(cfg.batch_size):
                    v = self.caps2[batch_size][self.argmax_idx[batch_size], :]
                    masked_v.append(tf.reshape(v, shape=(1, 1, 16, 1)))

                self.masked_v = tf.concat(masked_v, axis=0)
                assert self.masked_v.get_shape() == [cfg.batch_size, 1, 16, 1]
            else:
                self.masked_v = tf.multiply(tf.squeeze(self.caps2), tf.reshape(self.Y, (-1, 10, 1)))
                self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2), axis=2, keep_dims=True) + epsilon)

        # Reconstruct the images
        with tf.variable_scope('Decoder'):
            vector_j = tf.reshape(self.masked_v, shape=(cfg.batch_size, -1))
            fc1 = tf.contrib.layers.fully_connected(vector_j, num_outputs=512)
            assert fc1.get_shape() == [cfg.batch_size, 512]
            fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
            assert fc2.get_shape() == [cfg.batch_size, 1024]
            self.decoded = tf.contrib.layers.fully_connected(fc2, num_outputs=784, activation_fn=tf.sigmoid)

    def loss(self):
        # margin loss
        max_l = tf.square(tf.maximum(0., cfg.m_plus - self.v_length))
        max_r = tf.square(tf.maximum(0., self.v_length - cfg.m_minus))

        assert max_l.get_shape() == [cfg.batch_size, 10, 1, 1]

        max_l = tf.reshape(max_l, shape=(cfg.batch_size, -1))
        max_r = tf.reshape(max_r, shape=(cfg.batch_size, -1))

        T_c = self.Y

        L_c = T_c * max_l + cfg.lambda_val * (1 - T_c) * max_r

        self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

        # reconstruction loss
        origin = tf.reshape(self.X, shape=(cfg.batch_size, -1))
        squared = tf.square(self.decoded - origin)
        self.reconstruction_loss = tf.reduce_mean(squared)

        self.total_loss = self.margin_loss + cfg.regularization_scale*self.reconstruction_loss

    def _summary(self):
        train_summary = []
        train_summary.append(tf.summary.scalar('train/margin_loss', self.margin_loss))
        train_summary.append(tf.summary.scalar('train/reconstruction_loss', self.reconstruction_loss))
        train_summary.append(tf.summary.scalar('train/total_loss', self.total_loss))
        recon_img = tf.reshape(self.decoded, shape=(cfg.batch_size, 28, 28, 1))
        train_summary.append(tf.summary.image('reconstruction_img', recon_img))
        self.train_summary = tf.summary.merge(train_summary)

        correct_prediction = tf.equal(tf.to_int32(self.labels), self.argmax_idx)
        self.batch_accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
        self.test_acc = tf.placeholder_with_default(tf.constant(0.), shape=[])
        
