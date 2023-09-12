## Acting in Delayed Environments with Non-Stationary Markov Policies
This repository contains the implementation of the Delayed, Agumented, Oblivious, and RNN agents from the paper:
"[Acting in Delayed Environments with Non-Stationary Markov Policies](https://arxiv.org/pdf/2101.11992)", Esther Derman<sup>\*</sup>, Gal Dalal<sup>\*</sup>, Shie Mannor (<sup>*</sup>equal contribution), published in ICLR 2021. 

<img src="https://github.com/galdl/rl_delay_basic/blob/master/delayed_q_diagram.png" width="600" height="330">

The agent here supports the Atari environments. The simpler agent that supports Cartpole and Acrobot can be found [here](https://github.com/galdl/rl_delay_basic).

## Installation
This is a fork of [Stable-Baselines](https://github.com/hill-a/stable-baselines/releases/tag/v2.10.1) (v2.10.1, based on TensorFlow), with the addition of the delayed agent. 

To set up the environment please follow the instructions in [Stable-Baselines](https://github.com/hill-a/stable-baselines/releases/tag/v2.10.1).

## Running the code
Running the code is straightforward using run_experiment_rl_delay.py.

## Citing the Project

To cite this repository in publications:

```
@article{derman2021acting,
  title={Acting in delayed environments with non-stationary markov policies},
  author={Derman, Esther and Dalal, Gal and Mannor, Shie},
  journal={International Conference on Learning Representations (ICLR)},
  year={2021}
}
```
