from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf

def predictions_and_gradients_fn(mdl):
  '''
  :param inp: nBatch x nInput tensor
  :param target_label_index: int index of target label
  :return:
     predictions: nBatch x nClasses
     gradients: same shape as inp
         (gradient of target_label_index prediction w.r.t input)
  '''
  def fn(inp, target_label_index):
    x = tf.Variable(inp)
    with tf.GradientTape() as tape:
      predictions = tf.reshape(mdl(x)[:,target_label_index], [-1,1])
    grads = tape.gradient(predictions, x)
    return predictions, grads
  return fn

# basic approx IG code from https://github.com/ankurtaly/Integrated-Gradients
def integrated_gradients(inp,
                         target_label_index,
                         predictions_and_gradients,
                         baseline,
                         steps=50):

  """Computes integrated gradients for a given network and prediction label.

  Integrated gradients is a technique for attributing a deep network's
  prediction to its input features. It was introduced by:
  https://arxiv.org/abs/1703.01365

  In addition to the integrated gradients tensor, the method also
  returns some additional debugging information for sanity checking
  the computation. See sanity_check_integrated_gradients for how this
  information is used.

  This method only applies to classification networks, i.e., networks
  that predict a probability distribution across two or more class labels.

  Access to the specific network is provided to the method via a
  'predictions_and_gradients' function provided as argument to this method.
  The function takes a batch of inputs and a label, and returns the
  predicted probabilities of the label for the provided inputs, along with
  gradients of the prediction with respect to the input. Such a function
  should be easy to create in most deep learning frameworks.

  Args:
    inp: The specific input for which integrated gradients must be computed.
    target_label_index: Index of the target class for which integrated
    gradients
      must be computed.
    predictions_and_gradients: This is a function that provides access to the
      network's predictions and gradients. It takes the following
      arguments:
      - inputs: A batch of tensors of the same same shape as 'inp'. The first
          dimension is the batch dimension, and rest of the dimensions coincide
          with that of 'inp'.
      - target_label_index: The index of the target class for which gradients
        must be obtained.
      and returns:
      - predictions: Predicted probability distribution across all classes
          for each input. It has shape <batch, num_classes> where 'batch' is
          the
          number of inputs and num_classes is the number of classes for the
          model.
      - gradients: Gradients of the prediction for the target class (denoted by
          target_label_index) with respect to the inputs. It has the same shape
          as 'inputs'.
    baseline: [optional] The baseline input used in the integrated
      gradients computation. If None (default), the all zero tensor with
      the same shape as the input (i.e., 0*input) is used as the baseline.
      The provided baseline and input must have the same shape.
    steps: [optional] Number of intepolation steps between the baseline
      and the input used in the integrated gradients computation. These
      steps along determine the integral approximation error. By default,
      steps is set to 50.

  Returns:
    integrated_gradients: The integrated_gradients of the prediction for the
      provided prediction label to the input. It has the same shape as that of
      the input.

    The following output is meant to provide debug information for sanity
    checking the integrated gradients computation.
    See also: sanity_check_integrated_gradients

    prediction_trend: The predicted probability distribution across all classes
      for the various (scaled) inputs considered in computing integrated
      gradients.
      It has shape <steps, num_classes> where 'steps' is the number of
      integrated
      gradient steps and 'num_classes' is the number of target classes for the
      model.
  """
  if baseline is None:
    baseline = 0 * inp
  assert (baseline.shape == inp.shape)

  # Scale input and compute gradients.
  scaled_inputs = [baseline + (float(i) / steps) * (inp - baseline) for i in
                   range(0, steps + 1)]
  predictions, grads = predictions_and_gradients(scaled_inputs,
                                                 target_label_index)  #
  # shapes: <steps+1>, <steps+1, inp.shape>

  avg_grads = np.average(grads[:-1], axis=0)
  integrated_gradients = (inp - baseline) * avg_grads  # shape: <inp.shape>
  return integrated_gradients, predictions


def integrated_gradients_multi(inp,
                         target_label_index,
                         predictions_and_gradients,
                         baseline,
                         steps=50):

  """Computes integrated gradients for a given network, given batch of inputs,
  and corresponding prediction labels.

  Integrated gradients is a technique for attributing a deep network's
  prediction to its input features. It was introduced by:
  https://arxiv.org/abs/1703.01365

  In addition to the integrated gradients tensor, the method also
  returns some additional debugging information for sanity checking
  the computation. See sanity_check_integrated_gradients for how this
  information is used.

  This method only applies to classification networks, i.e., networks
  that predict a probability distribution across two or more class labels.

  Access to the specific network is provided to the method via a
  'predictions_and_gradients' function provided as argument to this method.
  The function takes a batch of inputs and a label, and returns the
  predicted probabilities of the label for the provided inputs, along with
  gradients of the prediction with respect to the input. Such a function
  should be easy to create in most deep learning frameworks.

  Args:
    inp: The input batch for which integrated gradients must be computed.
    target_label_index: Indices of the target class corresponding to
      the input batch, for which integrated gradients must be computed.
    predictions_and_gradients: This is a function that provides access to the
      network's predictions and gradients. It takes the following
      arguments:
      - inputs: A batch of tensors of the same same shape as 'inp'. The first
          dimension is the batch dimension, and rest of the dimensions coincide
          with that of 'inp'.
      - target_label_index: The index of the target class for which gradients
        must be obtained.
      and returns:
      - predictions: Predicted probability distribution across all classes
          for each input. It has shape <batch, num_classes> where 'batch' is
          the
          number of inputs and num_classes is the number of classes for the
          model.
      - gradients: Gradients of the prediction for the target class (denoted by
          target_label_index) with respect to the inputs. It has the same shape
          as 'inputs'.
    baseline: [optional] The baseline input used in the integrated
      gradients computation. If None (default), the all zero tensor with
      the same shape as the input (i.e., 0*input) is used as the baseline.
      The provided baseline and input must have the same shape.
    steps: [optional] Number of intepolation steps between the baseline
      and the input used in the integrated gradients computation. These
      steps along determine the integral approximation error. By default,
      steps is set to 50.

  Returns:
    integrated_gradients: The integrated_gradients of the prediction for the
      provided prediction label to the input. It has the same shape as that of
      the input.

    The following output is meant to provide debug information for sanity
    checking the integrated gradients computation.
    See also: sanity_check_integrated_gradients

    prediction_trend: The predicted probability distribution across all classes
      for the various (scaled) inputs considered in computing integrated
      gradients.
      It has shape <steps, num_classes> where 'steps' is the number of
      integrated
      gradient steps and 'num_classes' is the number of target classes for the
      model.
  """
  if baseline is None:
    baseline = 0 * inp
  assert (baseline.shape == inp.shape)

  # Scale input and compute gradients.
  scaled_inputs = [baseline + (float(i) / steps) * (inp - baseline) for i in
                   range(0, steps + 1)]
  predictions, grads = predictions_and_gradients(scaled_inputs,
                                                 target_label_index)  #
  # shapes: <steps+1>, <steps+1, inp.shape>

  avg_grads = np.average(grads[:-1], axis=0)
  integrated_gradients = (inp - baseline) * avg_grads  # shape: <inp.shape>
  return integrated_gradients, predictions

def ig_mtx(mdl, inp, steps=200):
  '''
  Compute IG matrix for input x = inp
  i.e. if y is final output vector, the returned matrix will
  have entry (i,j) equal to the IG (Delta x_i -> Delta y_j),
  where Delta x is the change in x from baseline to the current inp,
  and Delta y is the corresponding change in output y
  :param mdl: a tf.keras model
  :param inp: input vector (n, )
  :return: tensor of shape (n,k) where k is the number of outputs
  '''
  k = mdl.output.shape[1]
  pred_grad_fn = predictions_and_gradients_fn(mdl)
  igs = tf.concat([tf.reshape(integrated_gradients(inp, i,
                                                   pred_grad_fn,
                                                   inp*0, steps)[0], [-1,1])
                   for i in range(k)], axis=1)
  return igs


def random_baseline_integrated_gradients(
  inp,
  target_label_index,
  predictions_and_gradients,
  steps=50,
  num_random_trials=10):

  all_intgrads = []
  for i in range(num_random_trials):
    intgrads, prediction_trend = integrated_gradients(
      inp,
      target_label_index=target_label_index,
      predictions_and_gradients=predictions_and_gradients,
      baseline=255.0 * np.random.random([224, 224, 3]),
      steps=steps)
    all_intgrads.append(intgrads)
  avg_intgrads = np.average(np.array(all_intgrads), axis=0)
  return avg_intgrads