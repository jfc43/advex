import tensorflow as tflow
tf = tflow.compat.v1
import copy
import numpy as np
import re
import pandas as pd
from pmlb.write_metadata import get_types
from pmlb import fetch_data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import os
import shutil
from scipy.stats import norm
import glob
from sklearn.model_selection import ParameterGrid
import boto3
import uuid



class Bunch(dict):
  def __init__(self, *args, **kwargs):
    super(Bunch, self).__init__(*args, **kwargs)
    self.__dict__ = self

  def dict(self):
    return self.__dict__

  def mod(self, *dicts, **kwds):
    '''
    Modify with additional params and return new conf, BUT leave original
    unchanged !!
    A sequence of zero or more dicts OR Config objects can be passed,
    followed by
    a collection keyword-arguments.
    Examples:
    --------
    conf.mod(a = 100, b = True, c = 'junk')
    d = dict(a = 10, b = False)
    d1 = dict(a = 100, b = False)
    c1 = Bunch(a = 100, b = False)
    conf.mod(d, d1, c, a = 100, b = False)
    '''
    if len(dicts) > 0:
      dicts = list(dicts)
      new = self
      for d in dicts:
        if type(d) != dict:
          d = d.get()  # MUST be a Config !
        new = new.mod(**d)
      return new.mod(**kwds)
    new = copy.deepcopy(self.dict())
    new.update(kwds)
    return Bunch(**new)


def get_param_grid(params: Bunch):
  grid_dict = params.grid
  return list(ParameterGrid(grid_dict))

def create_bucket_name(bucket_prefix):
  # The generated bucket name must be between 3 and 63 chars long
  return ''.join([bucket_prefix, str(uuid.uuid4())])

def create_bucket(bucket_prefix, s3_connection):
  session = boto3.session.Session()
  current_region = session.region_name
  bucket_name = create_bucket_name(bucket_prefix)
  bucket_response = s3_connection.create_bucket(
    Bucket=bucket_name,
    CreateBucketConfiguration={
      'LocationConstraint': current_region})
  print(bucket_name, current_region)
  return bucket_name, bucket_response

def df_column_specs(df, params:Bunch):
  '''
  From a dataframe get column_specs list
  :param df:
  :param params: Contains categorical/numerical column overrides, target
  :return:
  '''
  if not params:
    params = Bunch()
  target = params.get('target')
  df.describe()
  cols = list(df.columns)
  cols = [c.replace('?', '_') for c in cols]
  cols = [c.replace(' ', '_') for c in cols]
  df.columns = cols
  if target is None:
    if 'class' in cols:
      target = 'class'
    else:
      target = 'target'
  # CAUTION: ALL categorical COLUMNS Are assumed to be already
  # encoded with categorical indices.
  specs = []
  for col in cols:
    vals = np.array(df[col])
    if col == target:
      continue
    uniques = np.unique(vals)
    # Note: binary, discrete are treated as categorical
    if (params.get('cat_cols') and col in params.cat_cols) or \
      (np.max(uniques) == len(uniques) - 1 and \
        not isinstance(vals[0], np.float) and \
        not isinstance(vals[0], np.float32) and \
        not isinstance(vals[0], np.float64) and \
        col not in params.get('num_cols', [])):
      card = len(np.unique(vals))
      specs += [dict(name=col, type='cat', card=card)]
    else:
      col_min = np.min(vals).astype(np.float32)
      col_max = np.max(vals).astype(np.float32)
      specs += [dict(name=col, type='num', card=1, min=col_min, max=col_max )]
  return specs, target

def df_binarize_col(df, col, uniques, positives=[0]):
  if np.array_equal(np.sort(uniques), [0,1]):
    return df
  df[col] = 1*(df[col].apply(lambda x: x in positives))
  return df


def round_if_num(x, n=2):
  if type(x) == str:
    return x
  return np.round(x, n)

def df_to_dicts(df, round=2):
  # deal with column names that have a dot in them!!
  cols = list(df.columns)
  pat = re.compile(r'[/\\.-]') # "/" or "." or "-"
  cols_nobad = [pat.sub('_', c) for c in cols]
  nobad2bad= dict(zip(cols_nobad, cols))
  # ok because df is typically small (a few rows at most) when using this fn
  df_copy = df.copy(deep=True)
  df_copy.columns = cols_nobad
  return [{nobad2bad[col]: round_if_num(getattr(row, col), round)
           for col in df_copy}
             for row in  df_copy.itertuples()]

def df_simple(df):
  '''
  return only the simple fields of a dict
  :param df:
  :return:
  '''
  df_strings = df.loc[:,(df.applymap(type)==str).all(0)]
  df_nums_bools = df.select_dtypes(include=['bool', 'number'],
                                   exclude=[object])
  df_ = pd.concat([df_strings, df_nums_bools], axis=1)

  return df_


def make_column_specs(dataset):
  df = fetch_data(dataset)
  return df_column_specs(df)

def split_and_standardize(df, params: Bunch):
  '''
  CAUTION: assumes label/target is the RIGHTmost column!
  :param df: dataframe
  :param params: various params, column/target specs etc
  :return:
  '''
  std = params.get('std', True)

  col_spec, target = df_column_specs(df, params)
  indices = df.index
  train_indices, test_indices = train_test_split(indices)
  df_train = df.iloc[train_indices]
  df_test = df.iloc[test_indices]
  df_train_original = df_train.copy(deep=True)
  df_test_original = df_test.copy(deep=True)
  #TODO: CAUTION these indices ASSUME target is AFTER all the features!
  numeric_columns = [i for (i,c) in enumerate(col_spec) if c['type'] == 'num']

  # fit the scaler ONLY on train data!
  scaler = StandardScaler()
  if len(numeric_columns) > 0 and std:
    scaler.fit(df_train.iloc[:, numeric_columns])
    # now use fitted scaler on train and test
    df_train.iloc[:, numeric_columns] = \
      scaler.transform(df_train.iloc[:, numeric_columns])
    df_test.iloc[:, numeric_columns] = \
      scaler.transform(df_test.iloc[:, numeric_columns])
  return df_train, df_test, df_train_original, df_test_original, scaler

def make_sibling_dir(curr_file, dir):
  '''
  E.g. if curr_file is "/a/b/c/d.py" , and dir = "e"
  ====> "/a/b/e/"
  :param curr_file: str
  :param dir: str
  :return:
  '''
  par_par_dir = Path(os.path.realpath(curr_file)).parent.parent
  path = os.path.join(par_par_dir, dir)
  if not os.path.exists(path):
    os.makedirs(path)
  return path

def make_clear_dir(dir):
  if not os.path.exists(dir):
    os.makedirs(dir)
  shutil.rmtree(dir)
  tf.summary.FileWriterCache.clear()

def last_file(dir, pattern = "*"):
  '''
  :param dir: directory or glob
  :return:
  '''
  if os.path.isdir(dir):
    dir = dir + '/' + pattern
  return sorted(glob.glob(dir))[-1]

def sub_dict(d, keys):
  '''
  get a sub-dictionary of some keys
  :param d:
  :param keys:
  :return:
  '''
  return dict(zip(keys, [d[k] for k in keys]))

def la_sig(nc, kc, nt, kt):
  pc = kc / nc  # prob
  pt = kt / nt
  vc = pc * (1 - pc) / nc  # variance
  vt = pt * (1 - pt) / nt
  sd = np.sqrt(vc + vt)  # stdev of additive lift
  est = pt - pc
  return dict(sd=sd, diff=est, sig=100 * norm.cdf(np.abs(est) / sd))

def sub_dict_prefix(d, prefix):
  '''
  get a sub-dictionary of keys starting with prefix
  :param d:
  :param keys:
  :return:
  '''
  key_matches = [k for k,v in d.items() if k.startswith(prefix)]
  return sub_dict(d, key_matches)

def tf_feature_columns(df, params: Bunch):
  col_spec, _ = df_column_specs(df, params=params)
  my_feature_columns = []
  for s in col_spec:
    col = s['name']
    if s['type'] == 'num':
      my_feature_columns.append(tf.feature_column.numeric_column(key=col))
    else:
      card = s['card']
      my_feature_columns.append(
        tf.feature_column.indicator_column(
          tf.feature_column.categorical_column_with_identity(
            col,
            num_buckets=card)))
  return my_feature_columns

def pmlb_dataset_x_y(dataset):
  col_spec, target = make_column_specs(dataset)
  X, y = fetch_data(dataset, return_X_y=True, local_cache_dir='./')
  numeric_columns = [i for (i,c) in enumerate(col_spec) if c['type'] == 'num']
  if len(numeric_columns) > 0:
    scaler = StandardScaler()
    X[:, numeric_columns] = scaler.fit_transform(X[:, numeric_columns])
  y = y.astype(np.float32)
  return X, y, col_spec, target

def entropy(x):
  x = np.abs(np.array(x))
  epsilon = 1e-10
  probs = np.minimum(np.maximum(x / np.sum(x), epsilon), 1-epsilon)
  return -np.sum(probs * np.log(probs))

def tf_entropy(x):
  x = tf.abs(x)
  if len(x.shape) == 1:
    x_sum = tf.reduce_sum(x)
  else:
    x_sum = tf.reduce_sum(x, axis=1, keep_dims=True)
  probs = x / x_sum
  return tf.keras.losses.binary_crossentropy(probs, probs)

def num_above_relative_threshold(X, thresh=0.1):
  '''
  given a matrix X (?nB, nF) for each row compute the number of elements
  whose absolute value is at least 0.1 times the biggest abs value in that
  row, and get the average of these.
  :param x:
  :return: float
  '''
  abs_thresh = thresh * tf.reduce_max(tf.abs(X), axis=1, keep_dims=True)
  exceeds = tf.cast(tf.greater_equal(tf.abs(X), abs_thresh), tf.float32)
  return tf.reduce_sum(exceeds, axis=1)

def tf_numpy(x):
  if tf.executing_eagerly():
    return x.numpy()
  else:
    return tf.Session().run(x)

def randnsphere(m, d):
  '''
  Generate m random points on unit d-sphere centered at origin
  :param m: num points
  :param d: num dims
  :return: m x d array
  '''
  x = np.random.normal(0.0, 1.0, m * d).reshape(m, d)
  inv_lens = 1.0 / np.sum(x * x, axis=1).reshape([-1,1])
  return x * inv_lens

# loss(model, inputs, labels, robust=True)


# grad(model, inputs, labels, robust=True)
