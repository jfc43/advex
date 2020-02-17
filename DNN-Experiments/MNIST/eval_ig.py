import numpy as np
import math
import cv2
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils import integrated_gradients, softmax, entropy, pct1pct, gini
import json

from model import Model

config_filename = 'config.json'

with open(config_filename) as config_file:
  config = json.load(config_file)

os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu_device']

num_steps = config['IG_steps']
model_dir = config['model_dir']

log_dir = os.path.join(model_dir, 'eval_logs')

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = open(os.path.join(log_dir, 'log.txt'), 'w')

reference_image = np.zeros((28, 28, 1), dtype=np.float32)

if __name__=='__main__':
  mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

  X = mnist.test.images.reshape([-1, 28, 28, 1])
  y = mnist.test.labels
  num_eval_examples = X.shape[0]

  total_corr = 0.0
  total_ent = 0.0
  total_a1p = 0.0
  total_gini = 0.0

  with tf.Session() as sess:
    model = Model()
    saver = tf.train.Saver()
    checkpoint = tf.train.latest_checkpoint(model_dir)
    saver.restore(sess, checkpoint)

    for i in range(num_eval_examples):
      test_image = X[i]
      original_label = y[i]

      corr = sess.run(model.num_correct, feed_dict={model.input: [test_image], model.label: [original_label]})
      total_corr += corr

      IG = integrated_gradients(sess, reference_image, test_image, original_label, model, gradient_func='output_input_gradient', steps=num_steps)

      IG_vector = IG.flatten()

      ent = entropy(IG_vector)
      total_ent += ent

      a1p = pct1pct(IG_vector)
      total_a1p += a1p

      gini_v = gini(IG_vector)
      total_gini += gini_v

      log_file.write('%d %.4f %.4f %.4f\n'%(corr, ent, a1p, gini_v))

    acc = total_corr / num_eval_examples

    avg_ent = total_ent / num_eval_examples
    avg_a1p = total_a1p / num_eval_examples
    adv_gini = total_gini / num_eval_examples

    print('Accuracy: {:.2f}%'.format(100 * acc))
    print('Average Entropy: {:.2f}'.format(avg_ent))
    print('Average A1P: {:.2f}%'.format(100 * avg_a1p))
    print('Average Gini: {:.4f}'.format(adv_gini))

    log_file.close()
