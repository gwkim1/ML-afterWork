import tensorflow as tf
import tensorflow.layers as layers
import gym
import math

import numpy as np

from skimage import transform
from skimage import data, color
import scipy.misc as smp

from collections import deque

from PIL import Image

import argparse

import warnings

warnings.filterwarnings('ignore')

class AutoEncoder:
    def __init__(self, env_name, learning_rate=0.001, batch_size=16, frozen_model_dir="./models/",
                 power_img_size=10, power_latent_size=8, resize_shape=120, noise_factor=0.5, debug=2,
                 pretrain_length=100, max_cache_size=1000000, checkpoints=4, tensorboard_dir="scratch/cae"):

        self.env = gym.make(env_name)
        self.debug = debug
        self.architecture = [2 << exponent for exponent in range(power_img_size)]
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.pretrain_length = pretrain_length
        self.cache = deque(maxlen=max_cache_size)
        self.resize_shape = resize_shape
        self.frozen_model_dir = "{}{}.ckpt".format(frozen_model_dir, 'CAE_{}_{}'.format(env_name, pretrain_length))
        self.noise_factor = noise_factor
        self.checkpoints = checkpoints
        self.inputs_ = tf.placeholder(tf.float32, (None, *[resize_shape, resize_shape, 1]), name='inputs')
        self.targets_ = tf.placeholder(tf.float32, (None, *[resize_shape, resize_shape, 1]), name='targets')
        self._prepopulate()

        #TODO
        self.writer = tf.summary.FileWriter(tensorboard_dir)

        # encoder
        encoder = self.inputs_
        weights = []
        shapes = []
        for filters in reversed(self.architecture[power_latent_size:]):
            n_input = encoder.get_shape().as_list()[3]
            W = tf.Variable(
            tf.random_uniform([
                3,
                3,
                n_input, filters],
                -1.0 / math.sqrt(n_input),
                1.0 / math.sqrt(n_input)))
            b = tf.Variable(tf.zeros([filters]))
            conv = tf.nn.conv2d(encoder, W, strides=[1,2,2,1], padding='SAME')
            weights.append(W)
            shapes.append(encoder.get_shape().as_list())
            encoder = tf.add(conv, b)

        self.encoder = encoder
        weights.reverse()
        shapes.reverse()

        decoded = self.encoder
        # decoder
        for i, shape in enumerate(shapes):
            W = weights[i]
            b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
            de_conv = tf.nn.conv2d_transpose(decoded, W, tf.stack([
                tf.shape(self.inputs_)[0], shape[1], shape[2], shape[3]]), strides=[1,2,2,1], padding='SAME')
            decoded = tf.add(de_conv, b)

        decoded = tf.nn.sigmoid(decoded)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.targets_, logits=decoded)
        self.cost = tf.reduce_mean(loss)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
        self.network = decoded

        self.saver = tf.train.Saver()
        self.sess = tf.Session()

    def sample(self):
        cache_size = len(self.cache)
        index = np.random.choice(np.arange(cache_size), size=self.batch_size, replace=False)
        return np.array([self.cache[i] for i in index])

    def _prepopulate(self):
        for i in range(self.pretrain_length):
            self.env.reset()
            done = False

            while not done:
                next_state, _, done, _ = self.env.step(self.env.action_space.sample())
                processed_image = self.preprocess_image(next_state)
                self.cache.append(processed_image)

    def preprocess_image(self, frame):
        return transform.resize(color.rgb2gray(frame), [self.resize_shape, self.resize_shape, 1], anti_aliasing=True)

    def add_noise(self, frame):
        frame = frame + self.noise_factor * np.random.randn(*frame.shape)
        return np.clip(frame, 0., 1.)

    def train(self, epochs=10):
        self.sess.run(tf.global_variables_initializer())
        batches = int(np.ceil(len(self.cache) / self.batch_size))
        checkpoint = math.floor(epochs/self.checkpoints)
        for epoch in range(epochs):
            if epoch % checkpoint <= 1:
                save_path = self.saver.save(self.sess, self.frozen_model_dir)
                if self.debug > 0:
                    print('Saved model to {}\n'.format(save_path))

            avg_cost_epoch = 0.
            for i in range(batches):

                batch = self.sample()
                noisy_batch = self.add_noise(batch)
                
                if self.debug > 0:
                    print('Running batch {}...\n'.format(i,))

                if self.debug > 2:
                    idx = np.random.choice(range(len(batch))) - 1
                    image = np.reshape(batch[idx], (self.resize_shape, self.resize_shape))
                    noisy_im = np.reshape(noisy_batch[idx], (self.resize_shape, self.resize_shape))
                    img_to_show = smp.toimage(image)
                    noisy_img_to_show = smp.toimage(noisy_im)
                    print('Noisy input image...\n')
                    noisy_img_to_show.show()
                    print('Target image')
                    img_to_show.show()

                batch_cost, _ = self.sess.run([self.cost, self.optimize], feed_dict={self.inputs_: noisy_batch,
                                                                                     self.targets_: batch})
                if self.debug > 0:
                    print("Batch cost: {}...\n", batch_cost)

                avg_cost_epoch += batch_cost / batches
                
            if self.debug > 0:
                print("Epoch: {}/{}...".format(epoch + 1, epochs), "Training loss: {}".format(avg_cost_epoch))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convolutional Autoencoder for gym environments")

    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--frozen_model_dir', type=str, default='./models/')
    parser.add_argument('--tensorboard_dir', type=str, default='scratch/cae')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--power_img_size', type=int, default=10)
    parser.add_argument('--power_latent_size', type=int, default=8)
    parser.add_argument('--resize_shape', type=int, default=120)
    parser.add_argument('--noise_factor', type=float, default=0.01)
    parser.add_argument('--max_cache_size', type=int, default=10000)
    parser.add_argument('--pretrain_length', type=int, default=100)
    parser.add_argument('--checkpoints', type=int, default=4)

    parser.add_argument('--env_name', type=str, default='Breakout-v0')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--debug', type=int, default=1)

    args = vars(parser.parse_args())

    env_name = args.pop('env_name')
    epochs = args.pop('epochs')

    cae = AutoEncoder(env_name, **args)

    cae.train(epochs=epochs)


