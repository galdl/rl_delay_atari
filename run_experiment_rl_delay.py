import gym
import wandb
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import LnCnnPolicy
from stable_baselines.common.policies import CnnLnLstmPolicy
from stable_baselines import A2C
from stable_baselines.deepq.delayed_dqn import DelayedDQN
from stable_baselines.common.atari_wrappers import DelayWrapper, MaxAndSkipEnv, wrap_deepmind
from functools import partial
import numpy as np

AVERAGE_OVER_LAST_EP = 0.05

def make_delayed_env(config):
    env = gym.make(config.env_name)
    if config.deepmind_wrapper:
        env = wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=True, scale=True)
    else:
        env = MaxAndSkipEnv(env, skip=4)
    env = DelayWrapper(env, config.delay_value, config.clone_full_state)
    return env

AGENT_NAME = 'agent_'
import platform
if platform.system() == 'Darwin':
    import os
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

hyperparameter_defaults = dict(
    train_freq=4,
    exploration_initial_eps=1.0,
    learning_rate=0.0001,
    target_network_update_freq=1000,
    exploration_final_eps=0.001,
    seed=1,
    env_name='MsPacman-v0', #'MsPacman-v0',
    gamma=0.99,
    delay_value=5,
    buffer_size=50000,
    prioritized_replay=True,
    # fixed_frame_skip=True,
    clone_full_state=False,
    load_pretrained_agent=False,
    agent_type='delayed', #'delayed', 'augmented', 'oblivious', 'rnn'
    num_rnn_envs=4,
    deepmind_wrapper=False,
    total_timesteps=int(1e6)
)
# Pass your defaults to wandb.init
wandb.init(config=hyperparameter_defaults, project="stable_baselines_tf-rl_delay")
config = wandb.config

agent_full_name = wandb.run.id + '_' + AGENT_NAME
# Save a checkpoint every 1000 steps
# checkpoint_callback = CheckpointCallback(save_freq=30*1800, save_path='./logs/',
#                                          name_prefix=agent_full_name)
checkpoint_callback = None
if config.agent_type == 'rnn':
    env = DummyVecEnv([partial(make_delayed_env, config=config)])
    model = A2C(CnnLnLstmPolicy, env, verbose=1, gamma=config.gamma, tensorboard_log='')
else:
    env = make_delayed_env(config)
    if config.agent_type == 'delayed':
        is_delayed_agent = True
        is_delayed_augmented_agent = False
    elif config.agent_type == 'augmented':
        is_delayed_agent = False
        is_delayed_augmented_agent = True
    else: # 'oblivious'
        is_delayed_agent = False
        is_delayed_augmented_agent = False

    model = DelayedDQN(LnCnnPolicy, env, verbose=1, train_freq=config.train_freq, learning_rate=config.learning_rate,
                    double_q=True, target_network_update_freq=config.target_network_update_freq,
                gamma=config.gamma, prioritized_replay=config.prioritized_replay, exploration_initial_eps=config.exploration_initial_eps,
                exploration_final_eps=config.exploration_final_eps, delay_value=config.delay_value,
                       forward_model=env, buffer_size=config.buffer_size, load_pretrained_agent=config.load_pretrained_agent,
                       is_delayed_agent=is_delayed_agent, is_delayed_augmented_agent=is_delayed_augmented_agent)

_, episode_rewards = model.learn(total_timesteps=config.total_timesteps, callback=checkpoint_callback)
tot_ep_num = len(episode_rewards)
avg_over = round(tot_ep_num * AVERAGE_OVER_LAST_EP)
final_avg_score = np.mean(episode_rewards[-avg_over:])
wandb.log({'final_score': final_avg_score})

