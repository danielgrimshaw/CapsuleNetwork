import tensorflow as tf

flags = tf.app.flags

# hypers

## separate margin loss
flags.DEFINE_float('m_plus', 0.9, 'm plus')
flags.DEFINE_float('m_minus', 0.1, 'm minus')
flags.DEFINE_float('lambda_val', 0.5, 'weight of loss for absent digit classes')

## training
flags.DEFINE_integer('batch_size', 25, 'batch size')
flags.DEFINE_integer('epochs', 50, 'epochs')
flags.DEFINE_integer('iter_routing', 3, 'number of iterations in routing alg')
flags.DEFINE_boolean('mask_with_y', True, 'use the true label to mask out target capsule')

flags.DEFINE_float('stddev', 0.01, 'stddev for initializer')
flags.DEFINE_float('regularization_scale', 0.392, 'regularization coefficient for reconstruction loss')

# environment settings
flags.DEFINE_string('dataset', 'data/CIFaR-10', 'the path to the dataset')
flags.DEFINE_boolean('is_training', True, 'train or predict')
flags.DEFINE_integer('num_threads', 8, 'number of enqueueing threads')
flags.DEFINE_string('logdir', 'logs', 'logs directory')
flags.DEFINE_integer('train_sum_freq', 50, 'steps between saving training summaries')
flags.DEFINE_integer('test_sum_freq', 500, 'steps between saving test summaries')
flags.DEFINE_integer('save_freq', 3, 'epochs between each save')
flags.DEFINE_string('results', 'results', 'path to save results')

# Distributed training
flags.DEFINE_integer('num_gpu', 1, 'number of gpus for distributed training')
flags.DEFINE_integer('batch_size_per_gpu', 64, 'batch size on each gpu')
flags.DEFINE_integer('threads_per_gpu', 8, 'Number of preprocessing threads on each machine')

cfg = tf.app.flags.FLAGS
