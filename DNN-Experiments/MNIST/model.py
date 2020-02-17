"""
The model is adapted from the tensorflow tutorial:
https://www.tensorflow.org/get_started/mnist/pros
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class Model(object):
  def __init__(self):
    self.input = tf.placeholder(tf.float32, shape = [None, 28, 28, 1])
    self.label = tf.placeholder(tf.int64, shape = [None])
    self.label_ph = tf.placeholder(tf.int32,shape=())

    # first convolutional layer
    self.W_conv1 = self._weight_variable([5,5,1,32])
    self.b_conv1 = self._bias_variable([32])

    # second convolutional layer
    self.W_conv2 = self._weight_variable([5,5,32,64])
    self.b_conv2 = self._bias_variable([64])

    # first fully connected layer
    self.W_fc1 = self._weight_variable([7 * 7 * 64, 1024])
    self.b_fc1 = self._bias_variable([1024])

    # output layer
    self.W_fc2 = self._weight_variable([1024,10])
    self.b_fc2 = self._bias_variable([10])
    
    # all weights
    self.all_weights = [self.W_conv1, self.b_conv1,
                       self.W_conv2, self.b_conv2,
                       self.W_fc1, self.b_fc1,
                       self.W_fc2, self.b_fc2]

    self.output = self.forward(self.input)

    self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.label, logits=self.output)

    self.sum_loss = tf.reduce_sum(self.loss)
    
    self.xent = self.sum_loss

    self.pred = tf.argmax(self.output, 1)

    correct_prediction = tf.equal(self.pred, self.label)

    self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    self.loss_input_gradient = tf.gradients(self.loss, self.input)[0]
    self.output_input_gradient = tf.gradients(self.output[:,self.label_ph], self.input)[0]

  def forward(self,x_input):

    h_conv1 = tf.nn.relu(self._conv2d(x_input, self.W_conv1) + self.b_conv1)
    h_pool1 = self._max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(self._conv2d(h_pool1, self.W_conv2) + self.b_conv2)
    h_pool2 = self._max_pool_2x2(h_conv2)

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.W_fc1) + self.b_fc1)

    return tf.matmul(h_fc1, self.W_fc2) + self.b_fc2

  @staticmethod
  def _weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

  @staticmethod
  def _bias_variable(shape):
      initial = tf.constant(0.1, shape = shape)
      return tf.Variable(initial)

  @staticmethod
  def _conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

  @staticmethod
  def _max_pool_2x2( x):
      return tf.nn.max_pool(x,
                            ksize = [1,2,2,1],
                            strides=[1,2,2,1],
                            padding='SAME')
