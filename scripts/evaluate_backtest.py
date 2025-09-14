import argparse
from datetime import datetime

from stable_baselines3 import PPO

from moex_rl_agent.data_loader import load_daily_multi
from moex_rl_agent.env import MultiTickerEnv


def evaluate(model_path, symbols, board, start, end, window=20):
    df = load_daily_multi(symbols, board, start, end)
    env = MultiTickerEnv(df, window=window, feature_cols=["sma20", "macd", "rsi14"])
    model = PPO.load(model_path)
    obs = env.reset()
    done = False
    port_hist = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, info = env.step(action)
        port_hist.append(info["portfolio_value"])
        if done:
            break
    return env, port_hist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--symbols", nargs="+", default=["SBER", "GAZP", "LKOH"])
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-06-01")
    args = parser.parse_args()

    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    env, hist = evaluate(args.model, args.symbols, "TQBR", start, end)
    import matplotlib.pyplot as plt

    plt.plot(hist)
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Step")
    plt.ylabel("Portfolio Value")
    plt.show()
