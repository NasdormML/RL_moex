from datetime import datetime

import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from moex_rl_agent.data_loader import load_daily_multi
from moex_rl_agent.env import MultiTickerEnv

SYMBOLS = ["SBER", "GAZP", "LKOH"]
BOARD = "TQBR"
START = datetime(2020, 1, 1)
END = datetime(2025, 6, 1)
WINDOW = 20


def objective(trial):
    lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    n_steps = trial.suggest_categorical("n_steps", [128, 256, 512])
    batch_size = trial.suggest_categorical("batch_size", [64, 128])
    clip_range = trial.suggest_uniform("clip_range", 0.1, 0.3)
    gamma = trial.suggest_uniform("gamma", 0.95, 0.999)
    max_weight = trial.suggest_uniform("max_weight_per_asset", 0.4, 0.9)
    cooldown = trial.suggest_int("cooldown_steps", 1, 5)
    turnover_pen = trial.suggest_uniform("turnover_penalty", 0.0, 0.2)

    df = load_daily_multi(SYMBOLS, BOARD, START, END)
    env = DummyVecEnv(
        [
            lambda: MultiTickerEnv(
                df,
                window=WINDOW,
                max_weight_per_asset=max_weight,
                cooldown_steps=cooldown,
                turnover_penalty=turnover_pen,
                feature_cols=["sma20", "macd", "rsi14"],
            )
        ]
    )
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        clip_range=clip_range,
        gamma=gamma,
    )
    model.learn(total_timesteps=50_000)
    raw_env = env.envs[0]
    obs = raw_env.reset()
    done = False
    last_port = raw_env.port_hist[-1]
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, info = raw_env.step(action)
        if done:
            break
    return raw_env.port_hist[-1] / (last_port + 1e-9)


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=12)
    print("Best params:", study.best_params)
