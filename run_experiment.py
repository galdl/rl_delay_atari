import gym
import wandb
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy, LnCnnPolicy, CnnPolicy
from stable_baselines import DQN
from stable_baselines.common.atari_wrappers import make_atari
from stable_baselines.common.callbacks import CheckpointCallback
ENV_NAME = 'MsPacman-v0'
AGENT_NAME = 'agent_' + ENV_NAME
TOTAL_TIMESTEPS = int(50e6)


wandb.init(project="stable_baselines-dqn")

env = gym.make(ENV_NAME)
# env = make_atari('BreakoutNoFrameskip-v4')
agent_full_name = wandb.run.id + '_' + AGENT_NAME
# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(save_freq=30*1800, save_path='./logs/',
                                         name_prefix=agent_full_name)

model = DQN(LnCnnPolicy, env, verbose=1, train_freq=4, exploration_fraction=0.01, learning_rate=0.0001)
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)
model.save(agent_full_name)

# del model # remove to demonstrate saving and loading
#
# model = DQN.load("deepq_cartpole")
#
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()