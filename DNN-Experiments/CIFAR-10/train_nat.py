"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
import shutil
import sys
from timeit import default_timer as timer

import tensorflow as tf
import numpy as np

from model import Model
import cifar10_input

config_filename = 'nat_config.json'

with open(config_filename) as config_file:
  config = json.load(config_file)

# Setting up training parameters
tf.set_random_seed(config['tf_random_seed'])
np.random.seed(config['np_random_seed'])

max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
step_size_schedule = config['step_size_schedule']
weight_decay = config['weight_decay']
data_path = config['data_path']
momentum = config['momentum']
batch_size = config['training_batch_size']
model_dir = config['model_dir']
loss_func = config['loss_func']
gpu_device = config['gpu_device']

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device

# Setting up the data and the model
raw_cifar = cifar10_input.CIFAR10Data(data_path)
global_step = tf.contrib.framework.get_or_create_global_step()
model = Model(mode='train')

# Setting up the optimizer
boundaries = [int(sss[0]) for sss in step_size_schedule]
boundaries = boundaries[1:]
values = [sss[1] for sss in step_size_schedule]
learning_rate = tf.train.piecewise_constant(
    tf.cast(global_step, tf.int32),
    boundaries,
    values)
total_loss = model.mean_xent + weight_decay * model.weight_decay_loss
# total_loss = model.margin_loss + weight_decay * model.weight_decay_loss
train_step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(
    total_loss,
    global_step=global_step)

# Setting up the Tensorboard and checkpoint outputs
if not os.path.exists(model_dir):
  os.makedirs(model_dir)

# We add accuracy and xent twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs

saver = tf.train.Saver(max_to_keep=3)
tf.summary.scalar('accuracy nat train', model.accuracy)
tf.summary.scalar('accuracy nat', model.accuracy)
tf.summary.scalar('xent nat train', model.xent / batch_size)
tf.summary.scalar('xent nat', model.xent / batch_size)
tf.summary.image('images nat train', model.input)
merged_summaries = tf.summary.merge_all()

# keep the configuration file with the model for reproducibility
shutil.copy(config_filename, model_dir)

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth=True

with tf.Session(config = tf_config) as sess:
  # initialize data augmentation
  cifar = cifar10_input.AugmentedCIFAR10Data(raw_cifar, sess, model)

  # Initialize the summary writer, global variables, and our time counter.
  summary_writer = tf.summary.FileWriter(model_dir, sess.graph)

  curr_step = 0
  sess.run(tf.global_variables_initializer())

  training_time = 0.0
  # Main training loop
  for ii in range(curr_step, max_num_training_steps):
    x_batch, y_batch = cifar.train_data.get_next_batch(batch_size,
                                                       multiple_passes=True)

    # Compute Adversarial Perturbations
    nat_dict = {model.input: x_batch,
                model.label: y_batch}

    # Output to stdout
    if ii % num_output_steps == 0:
      nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
      nat_loss = sess.run(model.xent, feed_dict=nat_dict)
      print('Step {}:    ({})'.format(ii, datetime.now()))
      print('    training nat accuracy {:.4}%, loss {:.4}'.format(nat_acc * 100, nat_loss))
      if ii != curr_step:
        print('    {} examples per second'.format(
            num_output_steps * batch_size / training_time))
        training_time = 0.0
    # Tensorboard summaries
    if ii % num_summary_steps == 0:
      summary = sess.run(merged_summaries, feed_dict=nat_dict)
      summary_writer.add_summary(summary, global_step.eval(sess))

    # Write a checkpoint
    if ii % num_checkpoint_steps == 0:
      saver.save(sess,
                 os.path.join(model_dir, 'checkpoint'),
                 global_step=global_step)

    # Actual training step
    start = timer()
    sess.run(train_step, feed_dict=nat_dict)
    end = timer()
    training_time += end - start
