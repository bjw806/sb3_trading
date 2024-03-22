# module import

import warnings
import gymnasium as gym
import numpy as np
import torch
from gymnasium.envs.registration import register
from stable_baselines3 import DQN, PPO  # noqa
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent import MlpLstmPolicy
from stable_baselines3.ppo import MlpPolicy  # noqa
from tqdm import TqdmExperimentalWarning
from preprocess import preprocess

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

register(
    id="MultiDatasetDiscretedTradingEnv",
    entry_point="gte:MultiDatasetDiscretedTradingEnv",
    disable_env_checker=True,
)


if __name__ == "__main__":
    env = gym.make(
        "MultiDatasetDiscretedTradingEnv",
        dataset_dir="./data/train/month/**/*.pkl",
        preprocess=preprocess,
        # reward_function=reward_only_position_changed,
        positions=[-10, 0, 10],
        trading_fees=0.0001 / 1000,
        borrow_interest_rate=0.000003,
        portfolio_initial_value=100,
        verbose=2,
        window_size=60,
    )
    env = gym.wrappers.FlattenObservation(env)

    model = RecurrentPPO(
        MlpLstmPolicy,
        env,
        # buffer_size=30000000,  # 1000000
        # batch_size=128,
        verbose=0,
        tensorboard_log="./tensorboard/",
        device="cpu",
        seed=2414411,
    )

    model.learn(total_timesteps=1000000, progress_bar=True)
    model.save("./model/RPPO/1.zip")
