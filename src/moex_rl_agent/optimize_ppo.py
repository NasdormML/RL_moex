from datetime import datetime

import optuna
from stable_baselines3 import PPO

from moex_rl_agent.data_loader import load_daily
from moex_rl_agent.env import MOEXTradingEnv

SYMBOL = "SBER"
BOARD = "TQBR"
START = datetime(2020, 1, 1)
END = datetime(2025, 6, 1)
WINDOW = 20
TIMESTEPS = 50000


def objective(trial):
    lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    n_steps = trial.suggest_categorical("n_steps", [128, 256, 512, 1024])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    clip_range = trial.suggest_uniform("clip_range", 0.1, 0.3)
    gae_lambda = trial.suggest_uniform("gae_lambda", 0.8, 1.0)

    df = load_daily(SYMBOL, BOARD, START, END)
    env = MOEXTradingEnv(df, window_size=WINDOW)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        clip_range=clip_range,
        gae_lambda=gae_lambda,
        ent_coef=0.01,
        verbose=0,
    )
    model.learn(total_timesteps=TIMESTEPS, log_interval=0)
    obs = env.reset()
    rewards = []
    for _ in range(len(df) - WINDOW):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, info = env.step(action)
        rewards.append(info["portfolio_value"])
        if done:
            break

    return rewards[-1] / rewards[0]  # >1 лучше


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    print("Best hyperparams:", study.best_params)
