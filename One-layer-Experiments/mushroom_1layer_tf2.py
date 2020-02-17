#!/usr/bin/env python
# coding: utf-8
import logging
logging.basicConfig(level=logging.ERROR)
from pathlib import Path
import warnings
with warnings.catch_warnings():
  warnings.filterwarnings("ignore",category=FutureWarning)
  import tensorflow as tf
  from tensorflow import feature_column
  from tensorflow.keras import layers
  from tensorflow.keras.layers import Dense, Flatten, Conv2D
import platform
import numpy as np
import pandas as pd
import math
import sys
import os
import glob
import time
from advtrain.cleverhans import projected_gradient_descent
import yaml
#!pip install -q tensorflow==2.0.0-beta1
import argparse
from sklearn.model_selection import train_test_split
from advtrain import model_metrics
from advtrain import adversarial
from utils import Bunch, df_column_specs, get_param_grid
tf.random.set_seed(123)
np.random.seed(312)
from test_tube import Experiment
from test_tube import HyperOptArgumentParser
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
exp_path = 'exp/mushroom'





######### arg parsing ###########
parser = argparse.ArgumentParser()
parser.add_argument('--hparams', default='mushroom-dnn-adv.yaml',
                    type=str,
                    help='params grid to run exps on')
parser.add_argument('-q', '--quick', default=False,
                    action='store_true',
                    help='whether to do a quick test')
parser.add_argument('--out', type=str, help='results output file')

args = parser.parse_args(sys.argv[1:])
# log_codes = dict(e=tf.logging.ERROR,
#                  i=tf.logging.INFO,
#                  w=tf.logging.WARN,
#                  d=tf.logging.DEBUG)

if args.out is not None:
  exp_path = os.path.join(args.out, exp_path)

exp = Experiment(exp_path)

params = Bunch(yaml.load(open(args.hparams)))
grid = get_param_grid(params)
params.pop('grid')
# tf.logging.set_verbosity(log_codes.get(params.log.lower()[0],
#                                        tf.logging.ERROR))

# if platform.system() == 'Linux':
#   tf.compat.v1.disable_eager_execution()
#   print(f'******** TF EAGER MODE DISABLED on {platform.system()} *****')




def run_exp(params):
  exp.tag(params)
  URL = 'mushroom/all.csv'
  dataframe = pd.read_csv(URL)
  dataframe.head()

  specs, target = df_column_specs(dataframe, params=None)

  train, test = train_test_split(dataframe, test_size=0.2)
  train, val = train_test_split(train, test_size=0.2)
  print(len(train), 'train examples')
  print(len(val), 'validation examples')
  print(len(test), 'test examples')

  batch_size = params.batch or 32

  def df_to_dataset(dataframe, shuffle=True, batch_size=batch_size):
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
      ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

  # A utility method to create a tf.data dataset from a Pandas Dataframe

  feature_columns = []
  # for mushroom we know the cols are all categorical, so we're not being too
  # careful here
  for col in specs:
    feature_columns.append(
      feature_column.indicator_column(
        feature_column.categorical_column_with_identity(
          col['name'], col['card'])))


  feature_layer = tf.keras.layers.DenseFeatures(feature_columns)



  train_ds = df_to_dataset(train, batch_size=batch_size)
  val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
  test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)


# ## Create Model
# 

# In[20]:


  featurizer = tf.keras.Sequential([
      feature_layer
    ],
      name='featurizer'
  )

  l1 = params.l1 or 0.0
  inner_model_logits = tf.keras.Sequential([
    layers.Dense(1,
                 kernel_regularizer=tf.keras.regularizers.l1(l=l1),
                 kernel_initializer=tf.random_normal_initializer(),
                 bias_initializer=tf.random_normal_initializer())
  ],
      name='inner_model_logits'
  )

  logit_prob = tf.keras.Sequential([
      layers.Activation('sigmoid')
  ], name='logit_prob')


  features_prob = tf.keras.Sequential([
      inner_model_logits,
      logit_prob
  ])

  # try out the model
  example_batch, label_batch = next(iter(train_ds))
  features = featurizer(example_batch)
  logits = inner_model_logits(features)
  logit_prob(logits)[:4]
  features_prob(features[:4])

#  predictions = model(example_batch)[:,0]

  loss_object = tf.keras.losses.BinaryCrossentropy()

  if params.adam:
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
  else:
    optimizer = tf.keras.optimizers.Ftrl(learning_rate=0.01)

#  features_prob.compile(optimizer=optimizer, loss=loss_object)
  def inner_model_logits_2_class(x):
    logits = inner_model_logits(x)
    return tf.concat([-logits, logits], axis=1)


  features_perturbed = projected_gradient_descent(inner_model_logits_2_class,
                                                  features, 0.4, 0.2,
                                                  5, np.inf,
                                                  y=label_batch, targeted=False)

  input_dim = features_prob.layers[0].inputs[0].shape[1]
  inputs = tf.keras.Input(shape=(input_dim,))
  outputs = tf.keras.layers.Dense(1,
                                  activation='sigmoid',
                                  kernel_regularizer = tf.keras.regularizers.l1(l=l1),
                                  kernel_initializer = tf.random_normal_initializer(),
                                  bias_initializer = tf.random_normal_initializer())(inputs)
  features_prob_functional = tf.keras.Model(inputs=inputs, outputs=outputs)

  model = tf.keras.Sequential([
    featurizer,
    features_prob_functional
    # inner_model_logits,
    # logit_prob
  ])
  model(example_batch)[:4]

  def inner_model_probs_2_class(x):
    probs = features_prob_functional(x)
    return tf.concat([1-probs, probs], axis=1)



  def mdl_to_2_class(mdl):
    def fn(x):
      probs = mdl(x)
      return tf.concat([1-probs, probs], axis=1)
    return fn


  # test out adv perturbation, and closed form

  # features_perturbed_closed_form = adversarial.logistic_perturb(
  #   inner_model_logits, features, label_batch, 0.4)
  #
  # [features_perturbed_closed_form[0, :5], features_perturbed[0, :5]]



  train_loss = tf.keras.metrics.Mean(name='train_loss')
  train_auc = tf.keras.metrics.AUC(name='train_auc')
  train_acc = tf.keras.metrics.BinaryAccuracy(name='train_acc')

  test_loss = tf.keras.metrics.Mean(name='test_loss')
  test_auc = tf.keras.metrics.AUC(name='test_auc')
  test_acc = tf.keras.metrics.BinaryAccuracy(name='test_acc')


# In[26]:



  #@tf.function
  def train_step_perturb_input(examples, labels, eps, eps_step, num_pgd_steps):
    features = featurizer(examples)
    features_prob_functional(features) # dummy call to force weights to be
    # created
    # this is done outside the tape, but it's ok because we've ensured
    # there are no trainable vars in the featurizer (no embeddings!)
    features_perturbed = adversarial.logistic_perturb(
      features_prob_functional, features, labels, eps)
    with tf.GradientTape() as tape:
      predictions = features_prob_functional(features_perturbed)
      loss = loss_object(labels, predictions[:,0])
      loss += sum(features_prob_functional.losses)
    vars = featurizer.trainable_variables + \
           features_prob_functional.trainable_variables
    gradients = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(gradients, vars))
    train_loss(loss)
    train_auc(labels, tf.reshape(predictions,[-1]))
    train_acc(labels, tf.reshape(predictions,[-1]))

  def train_step_perturb_last_layer(examples, labels, eps,
                                    freeze_initial_layers=True):
    with tf.GradientTape() as tape:
      features = featurizer(examples)
      left, right = model_metrics.sub_models(inner_model_logits)
      penultimate_activations = left(features)

      right(penultimate_activations) # dummy call to force weights to be created
      d = penultimate_activations.shape[1]
      activations_perturbed = adversarial.logistic_perturb(right,
                                                        penultimate_activations, labels,
                                                        eps/math.sqrt(d))
      logits_perturbed = right(activations_perturbed)
      predictions = logit_prob(logits_perturbed)
      loss = loss_object(labels, predictions[:,0])
      loss += sum(features_prob.losses)
    if freeze_initial_layers:
      mdl = inner_model_logits.layers[1]
    else:
      mdl = inner_model_logits
    vars = mdl.trainable_variables # + featurizer.trainable_variables
    gradients = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(gradients, vars))

    train_loss(loss)
    train_auc(labels, tf.reshape(predictions,[-1]))

  # In[27]:


  #@tf.function
  def test_step(examples, labels):
    predictions = model(examples)[:,0]
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_auc(labels, predictions)
    test_acc(labels, predictions)


# In[28]:


# example_batch, label_batch = next(iter(train_ds))
# predictions = model(example_batch)[:,0]
# loss_object(label_batch, predictions)


# In[29]:

  def get_all_x_y(ds):
    batches = []
    y = []
    for examples, labels in ds:
      x = featurizer(examples)
      batches = batches + [x]
      y = y + [labels]
    return tf.concat(batches, axis=0), tf.concat(y, axis=0)

  def reset_states():
    train_loss.reset_states()
    train_auc.reset_states()
    train_acc.reset_states()
    test_loss.reset_states()
    test_auc.reset_states()
    test_acc.reset_states()

  EPOCHS = params.epochs or 20
  if args.quick:
    EPOCHS = 2
  eps = params.eps or 0.0  # 1.3 ?
  num_pgd_steps = params.pgd_steps or 5
  perturb_input = params.perturb_input
  clean_epochs = params.clean_epochs or 2
  if args.quick:
    clean_epochs = 1
  eps_step_factor = params.eps_step_factor or 5.0
  for epoch in range(EPOCHS):
    reset_states()
    if epoch < clean_epochs:
      eps_val = 0
    else:
      eps_val = eps
    eps_step = eps_val/eps_step_factor

    for images, labels in train_ds:
      if perturb_input:
        train_step_perturb_input(images, labels, eps_val, eps_step, num_pgd_steps)
      else:
        # perturb penultimate layer, using closed form for binary prob output
        train_step_perturb_last_layer(images, labels, eps_val,
                                      freeze_initial_layers=True)

    for test_images, test_labels in test_ds:
      test_step(test_images, test_labels)

    template = 'Eps={}, Epoch {}, Loss: {}, AUC: {}, ' \
               'Test Loss: {}, Test AUC: {}, Test AC: {}'
    if epoch % 5 == 0:
          print (template.format(eps_val, epoch+1,
                           train_loss.result(),
                           train_auc.result()*100,
                           test_loss.result(),
                           test_auc.result()*100,
                           test_acc.result()*100))

  # dummy examples to force model shapes to be set
  example_batch, label_batch = next(iter(train_ds))
  model(example_batch)

  model_file = 'mushroom_models/' + f'eps={eps}'

  #model.save(model_file, overwrite=True)
  model.save_weights(model_file + ".ckpt", overwrite=True)
  m = model_metrics.weight_metrics(model)
  print(f'weight metrics with eps={eps}:')
  print(m)

  x_test, y_test = get_all_x_y(test_ds)
  attribution_method = 'shap' if params.attribution_shap else 'ig'
  mdl_shap = features_prob_functional
  mdl_ig = inner_model_probs_2_class
  av_ig_pct1pct, ig_ent, av_gini, ig_results = \
    model_metrics.attribs_pct( mdl_shap, mdl_ig,
                               x_test,
                               y_test,
                               attribution_method=attribution_method)
  print(f'eps={eps}: IG_pct1pct={av_ig_pct1pct}')

  print(f'****done with eps={eps} ****')
  print(f'****************************')

  params_and_results = params.mod(dict(
    test_auc = np.round(test_auc.result()*100,2),
    test_acc = np.round(test_acc.result()*100,2),
    ig_ent = np.round(ig_ent, 2),
    gini = np.round(av_gini, 3),
    ig_1p = np.round(av_ig_pct1pct[0], 2)))
  print('*** logging ****')
  print(params_and_results.dict())
  exp.log(params_and_results.dict())
  exp.save()
  exp.close()
  return ig_results

ig_init_cols = ['correct', 'ent', 'a1p', 'gin']
n_init_cols = len(ig_init_cols)
for i, p in enumerate(grid):
  if args.quick and i > 2:
    break
  str_p = list(p.keys())[0] + '=' + str(list(p.values())[0])
  ig_results = run_exp(params.mod(p))
  exp_dir = exp.get_data_path(exp.name, exp.version)
  ig_file_pkl = os.path.join(exp_dir, str_p, 'results.pkl')
  ig_file_csv = os.path.join(exp_dir, str_p, 'results.csv')
  ig_cols = ['ig' + str(c) for c in range(ig_results.shape[1] - n_init_cols)]
  cols = ig_init_cols + ig_cols
  df_ig_results = pd.DataFrame(ig_results, columns=cols)
  df_ig_results = df_ig_results.astype(dict(correct='int32'))
  os.makedirs(Path(ig_file_pkl).parent,  exist_ok=True)
  df_ig_results.to_csv(ig_file_csv, sep=' ')
  df_ig_results.to_pickle(ig_file_pkl)


print('!!!!!!!  experiments done  !!!!')
print('visualize with tensorboard using: ')
print(f'tensorboard --logdir ./{exp_path}')


list_of_files = glob.glob(os.path.join(exp_path, 'default', '*/metrics.csv'))
# * means
latest_file = max(list_of_files, key=os.path.getctime)
latest_dir = os.path.dirname(latest_file)
pkl_file = os.path.join(latest_dir, 'metrics.pkl')

df = pd.read_csv(latest_file)
df.to_pickle(pkl_file)

# fig, (ax1, ax2) = plt.subplots(2)
# fig.suptitle('Test AUC, IG1PCT vs epsilon')
# ax1.plot(df.eps, df.test_auc)
# ax1.set(title='Test AUC', xlabel='epsilon')
# ax1.label_outer()
# ax2.plot(df.eps, df.ig_pct1pct)
# ax2.set(title='IG1PCT', xlabel='epsilon')
# ax2.label_outer()
# plt.show()
