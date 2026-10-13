# DQN on LunarLander: experiments

Independent TensorFlow/Keras DQN implementation used to run a series of controlled experiments on Gymnasium's LunarLander.

| # | Question | Notebook | Report | Status |
|---|---|---|---|---|
| 01 | How does the discount factor `gamma` affect learning? | [01_gamma_sweep](notebooks/01_gamma_sweep.ipynb) | [results](results/01_gamma_sweep/README.md) | done (1 seed) |

## Layout
- `notebooks/`: one notebook per experiment (`NN_<name>.ipynb`)
- `results/NN_<name>/`: figures, summary CSVs, small per-run result files, and a report
- Heavy artifacts (network weights, checkpoints) live on Google Drive in `dqn_lunar/NN_<name>/` and are not committed.

## Running
Open a notebook in Colab, e.g.
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mjsafy/dqn-lunarlander-experiments/blob/main/notebooks/01_gamma_sweep.ipynb)
