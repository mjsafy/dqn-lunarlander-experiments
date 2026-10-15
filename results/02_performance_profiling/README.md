# 02: Where does DQN training time go on a CPU, and how much faster can it be made?

Colab free tier (2 vCPU, Intel Xeon 2.2 GHz), TensorFlow 2.21, Gymnasium 1.3.0, LunarLander-v3.
The full write-up is the final analysis cell at the end of
[the notebook](../../notebooks/02_performance_profiling.ipynb).

## Main findings (all numbers measured on the machine above)
- The gamma sweep of experiment 01 (3.75M env steps) took 75.6 min, about 800 steps/s. A component profile predicts 75.7 min.
- Per env step the TF baseline spends ~55% in `act` (one `tf.function` call), ~32% in `learn`, ~11% in Box2D. The loop is call-overhead-bound, not compute-bound.

| Variant (1 process, 1 thread, epsilon 0.05) | steps/s | vs baseline |
|---|---|---|
| TF/Keras baseline | 792 | 1.00x |
| TF learn + NumPy act, weights copied in one transfer | 1732 | 2.19x |
| Whole DQN in NumPy | 3942 | 4.98x |

- Rejected: XLA (slower), PyTorch (1.5x on the learning step only), JAX (same as NumPy). GPU and vectorised environments were not measured.
- Two processes at once add only about 1.2x (NumPy) to 1.35x (TF + NumPy act) total throughput.

## NumPy DQN validation (same hyperparameters as experiment 01, 3 seeds)
NumPy gradients match TensorFlow's to 1.2e-7. All 6 runs solved the task (average 200 over 100 episodes).

| gamma | episodes to solve (seeds 0/1/2) | greedy eval mean | TF run, seed 0 |
|---|---|---|---|
| 0.995 | 591 / 525 / 560 | 190 / 208 / 248 | solved at 562, eval 188 |
| 0.99 | 636 / 583 / 720 | 154 / 215 / 139 | solved at 722, eval 198 |

Validated only at gamma 0.995 and 0.99, not for gamma <= 0.95. Each benchmark ran once on a shared CPU, so speed-up factors are approximate.

## Files
- `profile_report.json`, `profile_learn_report.json`: raw profiling output
- `numpy_dqn_validation.csv`, `numpy_dqn_validation_rewards.npz`: validation runs
- `scripts/`: the profiling scripts used
