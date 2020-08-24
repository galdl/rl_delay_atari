import gym
import wandb
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy, LnCnnPolicy, CnnPolicy
from stable_baselines import DQN
from stable_baselines.deepq.policy_iteration import PI
from stable_baselines.deepq.delayed_dqn import DelayedDQN
from stable_baselines.common.atari_wrappers import make_atari, DelayWrapper
from stable_baselines.common.callbacks import CheckpointCallback
# ENV_NAME = 'MsPacman-v0'
AGENT_NAME = 'agent_'# + ENV_NAME
import platform
if platform.system() == 'Darwin':
    import os
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

TOTAL_TIMESTEPS = int(2e6)

hyperparameter_defaults = dict(
    train_freq=4,
    exploration_initial_eps=1.0,
    learning_rate=0.0001,
    target_network_update_freq=1000,
    exploration_final_eps=0.001,
    seed=1,
    env_name='MsPacman-v0',
    gamma=0.99,
    delay_value=0,
    augment_state=False,
    buffer_size=50000,
)
# Pass your defaults to wandb.init
wandb.init(config=hyperparameter_defaults, project="stable_baselines_tf-rl_delay")
config = wandb.config

env = gym.make(config.env_name)
env = DelayWrapper(env, config.delay_value)
#TODO: check if using fixed 4-frame skip is better
# env = make_atari('BreakoutNoFrameskip-v4')
agent_full_name = wandb.run.id + '_' + AGENT_NAME
# Save a checkpoint every 1000 steps
# checkpoint_callback = CheckpointCallback(save_freq=30*1800, save_path='./logs/',
#                                          name_prefix=agent_full_name)
checkpoint_callback = None
# model = DQN(LnCnnPolicy, env, verbose=1, train_freq=config.train_freq, learning_rate=config.learning_rate,
#                 double_q=True, target_network_update_freq=config.target_network_update_freq,
#             gamma=config.gamma, prioritized_replay=True, exploration_initial_eps=config.exploration_initial_eps,
#             exploration_final_eps=config.exploration_final_eps)
model = DelayedDQN(LnCnnPolicy, env, verbose=1, train_freq=config.train_freq, learning_rate=config.learning_rate,
                double_q=True, target_network_update_freq=config.target_network_update_freq,
            gamma=config.gamma, prioritized_replay=True, exploration_initial_eps=config.exploration_initial_eps,
            exploration_final_eps=config.exploration_final_eps, delay_value=config.delay_value,
                   forward_model=env, buffer_size=config.buffer_size)


model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)
# model.save(agent_full_name)

# del model # remove to demonstrate saving and loading
#
# model = DQN.load("deepq_cartpole")
#
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()