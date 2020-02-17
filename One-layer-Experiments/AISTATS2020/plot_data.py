import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
if sys.version_info[0] < 3:
  from StringIO import StringIO
else:
  from io import StringIO

# mnist_lambda = StringIO('''
# lambda accuracy entropy A1P Gini
# 0.000100 0.9922 0.6750 0.1750 0.9088
# 0.001000 0.9892 0.6732 0.1739 0.9099
# 0.010000 0.9866 0.6693 0.1723 0.9126
# 0.100000 0.9672 0.6701 0.1725 0.9121
# 0.300000 0.9413 0.6650 0.1693 0.9143
# 0.500000 0.9290 0.6583 0.1658 0.9184
# 1.000000 0.8822 0.6652 0.1685 0.9148
# ''')
#
# df_mnist = pd.read_csv(mnist_lambda, sep=' ')
# df_mnist.to_pickle('AISTATS2020/mnist_lambda.pkl')
#
#
# mnist_eps = StringIO('''
# epsilon accuracy entropy A1P Gini
# 0.100000 0.9947 0.6652 0.1670 0.9148
# 0.200000 0.9891 0.6035 0.1306 0.9443
# 0.300000 0.9815 0.5075 0.0525 0.9695
# 0.400000 0.1135 0.4552 0.0503 0.1161
# ''')
#
# df = pd.read_csv(mnist_eps, sep=' ')
# df
# df.to_pickle('AISTATS2020/mnist_eps.pkl')
#
# #it's more informative to show:
# # x-axis = Gini
# # y-axis1 = accuracy of L1reg
# # y-axis2 = accuracy of AdvTrain
# # i.e. we don't show epsilon, lambda
# # this way, it can fit on ONE plot
#
# mnist = StringIO('''
# accuracy Gini model
# 0.9947 0.9148 A
# 0.9891 0.9443 A
# 0.9815 0.9695 A
# 0.9922 0.9088 L
# 0.9892 0.9099 L
# 0.9866 0.9126 L
# 0.9672 0.9121 L
# 0.9413 0.9143 L
# 0.9290 0.9184 L
# 0.8822 0.9148 L
# ''')
#
# fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=True)
# df = pd.read_csv(mnist, sep=' ')
# ax = sns.lineplot(x='Gini', y='accuracy', hue='model', ax=ax, data=df)
#
# #sns.relplot(x='Gini', y='accuracy', hue='type', data=df, kind='line')
# plt.show()
#
# ALL datasets together

all = StringIO('''
accuracy Gini model dataset
0.9947 0.9148 A MNIST
0.9891 0.9443 A MNIST
0.9815 0.9695 A MNIST
0.9922 0.9088 L MNIST
0.9892 0.9099 L MNIST
0.9866 0.9126 L MNIST
0.9672 0.9121 L MNIST
0.9413 0.9143 L MNIST
0.9290 0.9184 L MNIST
0.8822 0.9148 L MNIST
0.9006 0.8035 A Fashion-MNIST
0.8620 0.8064 A Fashion-MNIST
0.8404 0.8451 A Fashion-MNIST
0.8159 0.9078 A Fashion-MNIST
0.7328 0.9417 A Fashion-MNIST
0.9070 0.7483 L Fashion-MNIST
0.9079 0.7498 L Fashion-MNIST
0.8751 0.7563 L Fashion-MNIST
0.8091 0.7866 L Fashion-MNIST
0.7577 0.7876 L Fashion-MNIST
0.7481 0.7874 L Fashion-MNIST
0.7220 0.7969 L Fashion-MNIST
0.998 0.967 A Mushroom
0.985 0.977 A Mushroom
0.972 0.978 A Mushroom
0.973 0.978 A Mushroom
0.97 0.972 L Mushroom
0.97 0.974 L Mushroom
0.967 0.976 L Mushroom
0.973 0.978 L Mushroom
0.914 0.975 L Mushroom
0.9275 0.719 A Spambase
0.883 0.764 A Spambase
0.892 0.777 A Spambase
0.925 0.717 L Spambase
0.898 0.74 L Spambase
''')

#plt.figure()
dims = (6, 7)
fig, ax = plt.subplots(nrows=2, ncols=2, squeeze=True, figsize=dims)
df = pd.read_csv(all, sep=' ')
titles = ['MNIST', 'Fashion-MNIST', 'Mushroom', 'Spambase']
rows = [0, 0, 1, 1]
cols = [0, 1, 0, 1]
for i, t in enumerate(titles):
  ax_ij = ax[rows[i], cols[i]]
  sns.lineplot(y='Gini', x='accuracy', hue='model', style='model',
               markers=True,
               ax=ax_ij, data=df.query(f'dataset=="{t}"'))
  plt.sca(ax_ij)
  plt.tight_layout()
  ax_ij.set_title(t)

plt.savefig('/Users/pchalasani/Dbox/Git/robulin-paper/AISTATS2020/varplots'
            '.png')
plt.show()
