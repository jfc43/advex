# Concise Explanations
This project is for the paper: Concise and Stable Explanations using Adversarial Training. Some codes are from [MNIST Challenge](https://github.com/MadryLab/mnist_challenge).

## Preliminaries
It is tested under Ubuntu Linux 16.04.1 and Python 3.6 environment, and requries some packages to be installed:
* [Tensorflow](https://www.tensorflow.org/install)
* [scipy](https://github.com/scipy/scipy)
* [sklearn](https://scikit-learn.org/stable/)
* [numpy](http://www.numpy.org/)
* [scikit-image](https://scikit-image.org/docs/dev/install.html)
* [shap](https://github.com/slundberg/shap)

## Downloading Datasets
* [MNIST](http://yann.lecun.com/exdb/mnist/): included in Tensorflow.
* [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist): could be loaded using Tensorflow.

## Overview of the Code
### Running Experiments
* Before doing experiments, first edit config.json file to specify experiment settings. We provide some template config files like nat_config.json.json. You may need to run `mkdir models` before training models.
* train_nat.py: the script used to train NATURAL models.
* train_nat_l1_reg.py: the script used to train L1-norm Regularization models.
* train_adv.py: the script used to train Madry's models.
* eval_ig.py: the script used to evaluate the model using IG.
* eval_shap.py: the script used to evaluate the model using SHAP.
* test_IG_sparse.ipynb: ipython notebook used to present demo figures for IG.
* test_SHAP_sparse.ipynb: ipython notebook used to present demo figures for SHAP.
