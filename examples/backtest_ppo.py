from datetime import datetime

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from moex_rl_agent.data_loader import load_daily_multi
from moex_rl_agent.env import MultiTickerEnv


def backtest(symbols, board, start, end, window, model_path):
    df = load_daily_multi(symbols, board, start, end)
    env = MultiTickerEnv(df, window=window)
    model = PPO.load(model_path)

    obs = env.reset()
    hist = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, done, info = env.step(action)
        date = env.dates[env.t - 1]
        hist.append(
            {
                "date": date,
                **{f"w_{s}": float(a) for s, a in zip(env.symbols, action)},
                "portfolio_value": info["portfolio_value"],
                "returns": rew,
            }
        )

    res = pd.DataFrame(hist)
    total = res["portfolio_value"].iloc[-1] / res["portfolio_value"].iloc[0] - 1
    rets = res["portfolio_value"].pct_change().fillna(0)
    sharpe = rets.mean() / (rets.std() + 1e-9) * np.sqrt(252)
    maxdd = (1 - res["portfolio_value"] / res["portfolio_value"].cummax()).max()

    print(f"\nMulti backtest {symbols} {start.date()}–{end.date()}")
    print(f"Return: {total*100:.2f}%, Sharpe: {sharpe:.2f}, MaxDD: {maxdd*100:.2f}%")

    out_csv = f"bt_multi_{'_'.join(symbols).lower()}.csv"
    res.to_csv(out_csv, index=False)
    print(f"Saved backtest to {out_csv}")


if __name__ == "__main__":
    backtest(
        symbols=["SBER", "GAZP", "LKOH"],
        board="TQBR",
        start=datetime(2025, 1, 1),
        end=datetime(2025, 6, 1),
        window=20,
        model_path="ppo_multi.zip",
    )
