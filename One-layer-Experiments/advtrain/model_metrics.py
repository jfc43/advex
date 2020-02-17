from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from ig import approx
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
import shap

# thanks to https://github.com/oliviaguest/gini
def gini(array):
  """Calculate the Gini coefficient of a numpy array."""
  # based on bottom eq:
  # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
  # from:
  # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
  # All values are treated equally, arrays must be 1d:
  array = np.array(array, dtype=np.float64)
  array = np.abs(array.flatten())
  if np.amin(array) < 0:
    # Values cannot be negative:
    array -= np.amin(array)
  # Values cannot be 0:
  array += 0.0000001
  # Values must be sorted:
  array = np.sort(array)
  # Index per array element:
  index = np.arange(1, array.shape[0] + 1)
  # Number of array elements:
  n = array.shape[0]
  # Gini coefficient:
  return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

def pct1pct(x):
  x = np.abs(x)
  x_max = np.max(x)
  return 100.0 * np.sum(x >= 0.01 * x_max)/len(x)

def model_layers(mdl):
  '''
  recursively expand layers to a flat list
  :param mdl:
  :return: flat list of all model layers
  '''
  if hasattr(mdl, 'layers'):
    sub_layers = [model_layers(layer) for layer in mdl.layers]
    layers = [item for sublist in sub_layers for item in sublist]
    return layers
  else:
    return [mdl]

def sub_models(mdl, i=-1):
  '''
  produce 2 models, one from input to layer i,
  another from i to last.
  :param mdl:
  :param i:
  :return: two sub-models
  '''
  layers = model_layers(mdl)
  mdl_left = tf.keras.Sequential(layers[:i])
  mdl_right = tf.keras.Sequential(layers[i:])
  return mdl_left, mdl_right

def weight_metrics(mdl):
  '''
  Compute some metrics quantifying the sparseness of the weights of a model.
  :param mdl: tf.keras model
  :return: dict of metrics
  '''
  layers = model_layers(mdl)
  n_layers = len(layers)
  wts_l1 = 0
  n_wts = 0
  wts = np.array([])
  wt_pcts = []
  for i in range(n_layers):
    layer = layers[i]
    try:
      W_abs = tf.reshape(tf.abs(layer.weights[0]), [-1]).numpy()
      n_wts += len(W_abs)
      wts_l1 += np.sum(W_abs)
      wts = np.concatenate([wts, W_abs])
      wt_pcts += [pct1pct(W_abs)]
    except:
      continue

  return dict(
    wts_l1=wts_l1,
    n_wts=n_wts,
    wts_1pct=pct1pct(wts),
    wt_pcts=wt_pcts
  )

def plot_cdf(sample):
  ecdf = ECDF(sample)
  x = np.linspace(min(sample), max(sample))
  plt.step(x, ecdf(x))
  plt.show()

def entropy(x):
  x = np.abs(x)
  probs = np.maximum(x/np.sum(x), 1e-10)
  return -np.sum(probs * np.log(probs))/np.log(len(x))

# def grad_shap(model, examples):
#   # FIX THIS WHOLE FN
#   def attribute(self,
#                 V: np.ndarray,
#                 n_samples=50,
#                 batch_size=1024) -> np.ndarray:
#     n_batches = (len(V) - 1) // batch_size + 1
#     X, _ = self.databunch.as_numpy()
#     batches = [
#       gradient_shap(
#         self.model,
#         V[i * batch_size:(i + 1) * batch_size],
#         X,
#         n_samples,
#       ) for i in range(n_batches)
#     ]
#     return tf.concat(batches, 0).numpy()


def attribs_pct(model_shap, model_ig, examples, labels,
                max_examples=1000, pct=[1, 5, 10],
                attribution_method ='ig'):
  '''
  Compute mean of pct1pct measure of IG-vectors, for given model and
  set of examples and labels
  :param model: keras model; a callable
  :param examples: nbatch x ...
  :param labels: nbatch
  :param max_examples: upper limit of num examples
  :return:
  '''
  pct = np.array(pct)/100.0
  n_x = np.minimum(max_examples, examples.shape[0])
  sum_ig_pct_over_thresh = 0
  pred_grad_fn = approx.predictions_and_gradients_fn(model_ig)
  if attribution_method == 'shap':
    n_examples = examples.shape[0]
    n_samples = np.minimum(3*n_x, n_examples)
    background = (tf.reshape(examples[0, ...], [1,-1]) * 0.0).numpy()
    # background = tf.gather(examples,
    #                        np.random.choice(n_examples, n_samples,
    #                                         replace=False)).numpy()

    #background = examples[:100].numpy()
    explainer_shap = shap.DeepExplainer(model_shap, background)

  sum_entropies = sum_ginis = 0
  n_steps = 20
  results = []
  for i in range(n_x):
    x = examples[i, ...]
    target_label_index = labels[i]
    # we use this regardless of attrib method because we're getting the
    # model-predictions (preds) here regardless of attrib method
    attribs, preds = approx.integrated_gradients(x, 1,
                                             pred_grad_fn, baseline=x * 0,
                                             steps=n_steps)
    # for shap, override the ig attribs
    if attribution_method == 'shap':
      x_row = tf.reshape(x, [1,-1])
      attribs = explainer_shap.shap_values(x_row.numpy())
    attribs = np.abs(tf.reshape(attribs, [-1]))
    pred = preds[n_steps,0]
    max_ig = np.max(attribs)
    n_big_igs = np.array([np.sum(attribs >= p * max_ig) for p in pct])
    ig_pct_over_thresh = 100.0 * n_big_igs / len(attribs)
    sum_ig_pct_over_thresh += ig_pct_over_thresh
    ig_ent = entropy(attribs)
    ig_gin = gini(attribs)
    sum_entropies += ig_ent
    sum_ginis += ig_gin
    is_correct = 1*((pred.numpy() > 0.5) == target_label_index.numpy())
    ig_list = [np.round(a, 3) for a in list(attribs)]
    results += [[is_correct, np.round(ig_ent, 3),
                 np.round(ig_pct_over_thresh[0]/100.0, 3),
                 np.round(ig_gin, 3)] + ig_list]
  av_ig_pct1pct = sum_ig_pct_over_thresh / n_x
  av_entropy = sum_entropies / n_x
  av_gin = sum_ginis / n_x
  return av_ig_pct1pct, av_entropy, av_gin, np.array(results)


