from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from ig import approx
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
import shap

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
