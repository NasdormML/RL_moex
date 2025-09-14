from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from moex_rl_agent.data_loader import load_daily_multi
from moex_rl_agent.env import MultiTickerEnv

SYMBOLS = ["SBER", "GAZP", "LKOH"]
BOARD = "TQBR"
START = datetime(2020, 1, 1)
END = datetime(2025, 6, 1)
WINDOW = 20
TOTAL_TIMESTEPS = 200_000
MODEL_PATH = "ppo_moex.zip"


def make_env():
    df = load_daily_multi(SYMBOLS, BOARD, START, END)
    feature_cols = [
        "sma20",
        "boll_upper20",
        "boll_lower20",
        "macd",
        "macd_signal",
        "rsi14",
    ]
    env = MultiTickerEnv(
        df,
        window=WINDOW,
        feature_cols=feature_cols,
        max_weight_per_asset=0.5,
        cooldown_steps=2,
        min_trade_value=200.0,
        max_trade_pct=0.2,
        turnover_penalty=0.05,
        weight_smoothing_alpha=0.2,
    )
    return env


def main():
    env = DummyVecEnv([lambda: make_env()])
    model = PPO(
        "MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=256, batch_size=64
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
