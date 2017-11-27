import os
import scipy
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt

from config import cfg

def unpickle(f):
    with open(os.path.join(cfg.dataset, f), 'rb') as fo:
        dictionary = pickle.load(fo, encoding='bytes')
    return dictionary

def one_hot(vec, vals=10):
    n = len(vec)
    out = np.zeros((n, vals)).astype(np.int32)
    out[range(n), vec] = 1
    return out

def display_cifar(images, size):
    n = len(images)
    plt.figure()
    plt.gca().set_axis_off()
    im = np.vstack([np.hstack([images[np.random.choice(n)] for i in range(size)]) for i in range(size)])
    plt.imshow(np.squeeze(im))
    plt.show()

class CifarLoader():
    def __init__(self, source):
        self._source = source
        self._i = 0
        self.images = None
        self.labels = None
    
    def load(self):
        data = [unpickle(f) for f in self._source]
        images = np.vstack([d[b'data'] for d in data])
        n = len(images)
        self.images = images.reshape(n, 3, 32, 32).mean(1, keepdims=True).transpose(0, 2, 3, 1).astype(np.float32)
        self.labels = np.hstack([d[b'labels'] for d in data]).astype(np.int32)

#        print(self.images.shape)
#        display_cifar(self.images, 10)
        
        return self

    def next_batch(self, batch_size):
        x = self.images[selt._i:self._i+batch_size]
        y = self.labels[self._i:self._i+batch_size]
        self._i = (self._i + batch_size) % len(self.images)
        return x, y

class CifarDataManager():
    def __init__(self):
        self.train = CifarLoader(["data_batch_{}".format(i)
            for i in range(1, 6)]).load()
        self.test = CifarLoader(["test_batch"]).load()

def get_batch_data():
    cifar = CifarDataManager()
    trX, trY = cifar.train.images, cifar.train.labels

    data_queues = tf.train.slice_input_producer([trX, trY])
    X, Y = tf.train.shuffle_batch(data_queues, num_threads=cfg.num_threads,
            batch_size=cfg.batch_size, capacity=cfg.batch_size * 64,
            min_after_dequeue=cfg.batch_size * 32,
            allow_smaller_final_batch=False)
    return (X, Y)

def save_images(imgs, size, path):
    imgs = (imgs + 1.) / 2
    return (scipy.misc.imsave(path, mergeImgs(imgs, size)))

def mergeImgs(images, size):
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        imgs[j*h:j*h + h, i*w:i*w + w, :] = image

    return imgs

if __name__ == "__main__":
    X, Y = load_mnist(cfg.dataset, cfg.is_training)
    print(X.get_shape())
    print(X.dtype)
