import numpy as np
import tensorflow as tf

tf.random.set_seed(123)
np.random.seed(312)

def logistic_perturb(mdl, examples, labels, eps):
  '''
  :param mdl: keras model (just 1 layer)
  :param examples: nbatch x ...
  :param labels: 0/1 labels tensor of shape (nbatch,)
  :return: L_inf(eps)-adversarially perturbed examples
  '''
  sign_y = tf.reshape(2*labels - 1, [-1,1]) # 0 -> -1, 1 -> 1
  w = tf.squeeze(mdl.weights[0]) # should have same shape as a single example
  delta = -eps * tf.sign(w) * tf.cast(sign_y, tf.float32)
  return examples + delta


def penultimate_softmax_perturb(mdl, examples, labels, eps):
  '''
  Perturb batch of examples with Near-optimal delta* subject to L_inf norm
  bound of eps. This is a closed form expression does not need gradients,
  so it helps with custom-training where we need to do this within the
  GradientTape clause.
  CAUTION - we found this approximation gets bad at larger epsilons,
  so this is DEPRECATED. The best way is to still use PGD (e.g. from
  cleverhans).
  :param mdl: keras model of penultimate layer (i.e. linear + softmax)
  :param examples: nbatch x ...: input to penultimate layer
  :param labels: true label indices (nbatch)
  :param eps: L_inf norm bound of delta perturbation
  :return:
  '''
  n_x = examples.shape[0]
  probs = mdl(examples)
  W = mdl.weights[0]
  y_label_column = tf.cast(tf.reshape(labels, [-1,1]), tf.int64)
  W_cols = tf.transpose(tf.gather_nd(tf.transpose(W),
                                     y_label_column))
  probs_indices = tf.concat([tf.reshape(np.arange(n_x), [-1,1]),
                             y_label_column], axis=1)
  label_probs = tf.gather_nd(probs, probs_indices)
  gradx = W_cols - label_probs * (tf.matmul(W, tf.transpose(probs)))
  delta_closed_form = -tf.sign(gradx) * eps
  return examples + tf.transpose(delta_closed_form)
