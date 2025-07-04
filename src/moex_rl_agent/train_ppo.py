import argparse
from datetime import datetime

from stable_baselines3 import PPO

from moex_rl_agent.data_loader import load_daily_multi
from moex_rl_agent.env import MultiTickerEnv


def main(symbols, board, start, end, window, timesteps, out):
    df = load_daily_multi(symbols, board, start, end)
    env = MultiTickerEnv(df, window=window)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save(out)
    print(f"Multi-ticker PPO saved to {out}.zip")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=["SBER", "GAZP", "LKOH"])
    p.add_argument("--board", default="TQBR")
    p.add_argument(
        "--start", type=lambda s: datetime.fromisoformat(s), default="2020-01-01"
    )
    p.add_argument(
        "--end", type=lambda s: datetime.fromisoformat(s), default="2024-12-31"
    )
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--timesteps", type=int, default=200000)
    p.add_argument("--out", default="ppo_multi")
    args = p.parse_args()

    main(
        symbols=args.symbols,
        board=args.board,
        start=args.start,
        end=args.end,
        window=args.window,
        timesteps=args.timesteps,
        out=args.out,
    )
