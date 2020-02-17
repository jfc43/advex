from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf


def ig_softmax(x, baseline=None):
  '''
  Analytic computation of IG-matrix for a pure-softmax layer
  :param x_k_1: input of shape (, k)
  :return:
     tensor (k x k) of IG of Delta x_i -> Delta p_j
         where p is the final probability vector after softmax.
     dp: (k x 1) change of final probs from baseline to input x
  '''
  x_k_1 = tf.reshape(x, [-1, 1]) # force to be column vec
  k = x_k_1.shape[0]
  if baseline is None:
    baseline = x_k_1 * 0
  baseline = tf.reshape(baseline, [-1,1])
  v_k_1 = tf.exp(x_k_1)
  v0 = tf.exp(baseline)
  dv_k_1 = v_k_1 - v0
  p_k_1 = v_k_1 / tf.reduce_sum(v_k_1)
  p0_k_1 = v0 / tf.reduce_sum(v0)
  dp_k_1 = p_k_1 - p0_k_1
  h = tf.reduce_sum(dv_k_1)
  hp_k_1 = h - dv_k_1  # h' or "h prime"
  A_k_k = tf.matmul(dv_k_1, tf.transpose(dp_k_1)) / h
  L_k_1 = tf.math.log(tf.abs(dv_k_1 - p_k_1 * h))
  L0_k_1 = tf.math.log(tf.abs(dv_k_1 - p0_k_1 * h))
  dL_k_1 = L_k_1 - L0_k_1
  B_k_k = (1 / (h * h)) * dv_k_1 * (
    -tf.linalg.diag(tf.squeeze(hp_k_1 * dL_k_1)) +
    (1 - tf.eye(k)) * tf.transpose(dv_k_1 * dL_k_1))
  return A_k_k + B_k_k, dp_k_1



def ig_1_1(du, W, dz, fraction=False):
  '''
  Compute IG-matrix for a generic layer with a "1-1" activation, i.e.
  an activation fn that maps each element of a vector to corresponding value
  (i.e. does not "mix" values like softmax does).
  Note that all the typical INTERMEDIATE layer activations are 1-1,
  e.g. sigmoid, RELU, tanh, etc. The picture is:

  u --W--> (v) --A--> z

  where W is the weight matrix, v = W^T u,
  A is the 1-1 activation fn mapping v to z

  :param du_k_1: change in input layer value (k x 1)
  :param W: weight matrix (k x m)
  :param dz_m_1: change in final activation (m x 1)
  :param fraction: whether to return fractional IG
  :return: IG matrix (k x m)
  '''
  # force du, dz to be row vecs
  du_1_k = tf.reshape(du, [1, -1])
  dz_1_m = tf.reshape(dz, [1, -1])
  frac_ig =  (tf.transpose(du_1_k) * W) / tf.matmul(du_1_k, W)
  if fraction:
    return frac_ig
  return dz_1_m * frac_ig


def ig_combine(g1, dv, g2):
  '''
  Combine IG-matrices of two adjacent sub-networks.
  g1 is the IG of the left sub-net whose last layer activations change by dv
  g2 is the IG of the right sub-net whose leftmost layer-change is dv
  :param g1: IG-matrix of left sub-net (k x m)
  :param dv: Change in activations at layer connecting the left, right nets
             (m x 1)
  :param g2: IG-matrix of left sub-net (m x n)
  :return:
  '''
  dv_1_m = tf.reshape(dv, [1,-1])
  return tf.matmul(g1 / dv_1_m, g2)

def ig_dense_softmax(x, W, bias, baseline=None):
  '''
  IG-matrix of layer with input x, dense-weights W, softmax activation
  :param x: len k input vec
  :param baseline: len k baseline vec
  :param W: weight matrix (k x m)
  :param bias: bias vector (1 x m)
  :return: IG-matrix (k x m)
  '''
  x = tf.reshape(x, [1,-1])
  bias = tf.reshape(bias, [1, -1])
  if baseline is None:
    baseline = x * 0
  dx = x - baseline
  z = tf.matmul(x, W) + bias
  z0 = tf.matmul(baseline, W) + bias
  dz = z - z0
  ig1 = ig_1_1(dx, W, dz)
  ig2, dp = ig_softmax(z, z0)
  return ig_combine(ig1, dz, ig2), dp

def ig_exact(mdl, inp, baseline=None):
  '''
  Compute IG-matrix of a keras model for inp relative to baseline
  :param mdl: tf.keras model
  :param inp: input tensor,  shape (k,)
  :param baseline: baseline input tensor, shape (k,)
  :return: IG matrix, shape k x m, where m = num outputs of model
  '''
  x_1_k = tf.reshape(inp, [1,-1])
  k = x_1_k.shape[1]
  if baseline is None:
    baseline = x_1_k * 0
  baseline = tf.reshape(baseline, [1,-1])
  n_layers = len(mdl.layers)
  ig = tf.eye(k)
  u = x_1_k
  u0 = baseline
  for i in range(n_layers):
    # layer is:
    # v ---(W,bias)---> v ---(activation)---> z
    # we don't need v explicitly
    layer = mdl.layers[i]
    W = layer.weights[0]
    bias = layer.weights[1]
    layer_fn = tf.keras.backend.function([layer.input],
                                         [layer.output])

    z = layer_fn([u])[0]
    z0 = layer_fn([u0])[0]
    # for all but last layer, we only want fractional IG, not total IG
    frac = (i < n_layers - 1)
    if layer.activation == tf.keras.activations.softmax:
      # in this case we MUST be at final layer
      assert(frac == False)
      layer_ig, _  = ig_dense_softmax(u, W, bias, u0)
    else:
      layer_ig = ig_1_1(u-u0, W, z - z0, frac)
    ig = tf.matmul(ig,  layer_ig)
    u = z
    u0 = z0
  return ig, z-z0


