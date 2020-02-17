# 6 Oct 2019
env PYTHONPATH=. python src/plot-vars.py  \
--params=plot-params/AISTATS2020/ig-l1-mnist-jc.yaml \
--file=AISTATS2020/mnist_lambda.pkl


# igpct1pct vs epsilon: perturb input layer
env PYTHONPATH=. python src/plot-vars.py  \
--params=plot-params/AISTATS2020/ig-eps-mushroom-dnn.yaml \
--file=/Users/pchalasani/Xbox/aistats-sep27/exp/mushroom/default/version_2

# igpct1pct vs epsilon: perturb penult (i.e "last") layer
# This needs to be re-done
#python mushroom_tf2.py --hparams=hparams/mushroom-dnn-adv-last.yaml --out aistats-sep27/mush-adv-last

env PYTHONPATH=. python src/plot-vars.py  \
--params=plot-params/AISTATS2020/ig-eps-mushroom-dnn-last.yaml \
--file=/Users/pchalasani/Xbox/aistats-sep21/exp/mushroom/default/version_1


# igpct1pct vs l1-reg(last layer):
# Strange that as l1 lambda increases, first IG-1pct dips to 1% then goes up
env PYTHONPATH=. python src/plot-vars.py  \
--params=plot-params/AISTATS2020/ig-eps-mushroom-dnn-l1.yaml \
--file=/Users/pchalasani/Xbox/aistats-sep26/exp/mushroom/default/version_1


#### MNIST #####

# Adv-input: vs eps
env PYTHONPATH=. python src/plot-vars.py  \
--params=plot-params/AISTATS2020/ig-eps-mnist-input-adv.yaml \
--file=plot-data/mnist-input-adv.pkl
#--file=/Users/pchalasani/Xbox/aistats-sep21/exp/mnist/default/version_4

# Adv input l1
env PYTHONPATH=. python src/plot-vars.py  \
--params=plot-params/AISTATS2020/ig-eps-mnist-input-l1.yaml \
--file=/Users/pchalasani/Xbox/aistats-sep21/exp/mnist/default/version_1

# Adv last
env PYTHONPATH=. python src/plot-vars.py  \
--params=plot-params/AISTATS2020/ig-eps-mnist-last-adv.yaml \
--file=/Users/pchalasani/Xbox/aistats-sep21/exp/mnist/default/version_2

# Adv last l1
env PYTHONPATH=. python src/plot-vars.py  \
--params=plot-params/AISTATS2020/ig-eps-mnist-last-l1.yaml \
--file=/Users/pchalasani/Xbox/aistats-sep21/exp/mnist/default/version_3







