import gym_trading_env  # noqa
import gymnasium as gym
from stable_baselines3 import A2C


def run1():
    env = gym.make(
        "MultiDatasetTradingEnv",
        dataset_dir="D:/Destktop/PyCharm_Projects/rl/data/train/month_1h/**/*.pkl",
        positions=[-1, 0, 1],
        trading_fees=0.01 / 100,
        borrow_interest_rate=(0.0003 / 100),
    )

    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000)

    vec_env = model.get_env()
    obs = vec_env.reset()

    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render("human")
        # VecEnv resets automatically
        # if done:
        #   obs = vec_env.reset()


if __name__ == "__main__":
    run1()
