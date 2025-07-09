from datetime import datetime

import optuna
from stable_baselines3 import PPO

from moex_rl_agent.data_loader import load_daily_multi
from moex_rl_agent.env import MultiTickerEnv


def objective(trial):
    # Подбор гиперпараметров с использованием Optuna
    lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    clip_range = trial.suggest_uniform("clip_range", 0.1, 0.3)
    gamma = trial.suggest_uniform("gamma", 0.9, 0.999)
    n_steps = trial.suggest_categorical("n_steps", [128, 256, 512])

    df = load_daily_multi(
        ["SBER", "GAZP", "LKOH"], "TQBR", datetime(2020, 1, 1), datetime(2024, 12, 31)
    )
    env = MultiTickerEnv(df, window=20)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=lr,
        batch_size=batch_size,
        clip_range=clip_range,
        gamma=gamma,
        n_steps=n_steps,
        verbose=1,
    )

    # Обучаем модель
    model.learn(total_timesteps=200000)
    return env.port_hist[-1]


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    print(f"Best hyperparameters: {study.best_params}")
