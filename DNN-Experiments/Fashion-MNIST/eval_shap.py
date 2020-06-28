import numpy as np
import math
import cv2
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils import softmax, gini
import shap
import json

from model import Model

config_filename = 'config.json'

with open(config_filename) as config_file:
  config = json.load(config_file)

os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu_device']

model_dir = config['model_dir']

log_dir = os.path.join(model_dir, 'eval_logs')

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = open(os.path.join(log_dir, 'shap_log.txt'), 'w')

reference_images = np.zeros((1, 28, 28, 1), dtype=np.float32)

if __name__=='__main__':
  fashion_mnist = input_data.read_data_sets('data/fashion', source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/', one_hot=False)

  X = fashion_mnist.test.images.reshape([-1, 28, 28, 1])
  y = fashion_mnist.test.labels
  num_eval_examples = X.shape[0]

  total_corr = 0.0
  total_gini = 0.0

  with tf.Session() as sess:
    model = Model()
    saver = tf.train.Saver()
    checkpoint = tf.train.latest_checkpoint(model_dir)
    saver.restore(sess, checkpoint)

    e = shap.DeepExplainer((model.input, model.output), reference_images)

    for i in range(num_eval_examples):
      test_image = X[i]
      original_label = y[i]

      corr = sess.run(model.num_correct, feed_dict={model.input: [test_image], model.label: [original_label]})
      total_corr += corr

      shap_value = e.shap_values(X[i:i+1])[0]

      shap_value_vector = shap_value.flatten()
      gini_v = gini(shap_value_vector)
      total_gini += gini_v

      log_file.write('%d %.4f\n'%(corr, gini_v))

    acc = total_corr / num_eval_examples
    avg_gini = total_gini / num_eval_examples

    print('Accuracy: {:.2f}%'.format(100 * acc))
    print('Average Gini: {:.4f}'.format(avg_gini))

    log_file.close()
