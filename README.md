# DQN on LunarLander: experiments

Independent DQN implementation (TensorFlow/Keras, plus a NumPy re-implementation in experiment 02) used to run a series of controlled experiments on Gymnasium's LunarLander.

| # | Question | Notebook | Report | Status |
|---|---|---|---|---|
| 01 | How does the discount factor `gamma` affect learning? | [01_gamma_sweep](notebooks/01_gamma_sweep.ipynb) | [results](results/01_gamma_sweep/README.md) | done (1 seed) |
| 02 | Where does DQN training time go on a CPU, and how much faster can it be made? | [02_performance_profiling](notebooks/02_performance_profiling.ipynb) | [results](results/02_performance_profiling/README.md) | done (validation: 3 seeds) |

## Layout
- `notebooks/`: one notebook per experiment (`NN_<name>.ipynb`)
- `results/NN_<name>/`: figures, summary CSVs, small per-run result files, and a report
- Heavy artifacts (network weights, checkpoints) live on Google Drive in `dqn_lunar/NN_<name>/` and are not committed.

## Running
Open a notebook in Colab, e.g.
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mjsafy/dqn-lunarlander-experiments/blob/main/notebooks/01_gamma_sweep.ipynb)
