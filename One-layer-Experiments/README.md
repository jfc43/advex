## Instructions to run experiments

Get python 3.7.3, create and activate a new venv, install pkgs:

```
python -m venv myenv
. ./myenv/bin/activate
pip install -r requirements.txt
```

To run the experiments for aistats look at the commands in `cmds-gcloud.sh`, 
and adjust the `--out` to where you want the results to be output. 
I use `test-tube` to save output results, typically they go into a path like   
`mushroom/default/version_0/metrics.csv`, which will contain various hyperparams 
and result metrics, including test-accuracy, AUC, IG sparseness measures, etc.

When an experiment is re-run with the same `--out` path, the results will be under 
a new version, e.g. `version_1`, `version_2`, etc.

Under a given version directory, there is a directory for 
each value of adversarial `eps`, e.g. `eps=0.1`, and in that directory 
there is a data-frame in `csv` and `pkl` format that contains 
various metrics per example, including label correctness, Gini index of the IG vector, etc.
Each row also has the complete IG vector.

Similarly for the experiments with varying L1-reg lambda, 
there is a directory for each value, e.g., `l1=0.1` etc with a similar data-frame. 

