import argparse
from datetime import datetime

from stable_baselines3 import PPO

from moex_rl_agent.data_loader import load_daily_multi
from moex_rl_agent.env import MultiTickerEnv


def evaluate(model_path, symbols, board, start, end, window, feature_cols):
    df = load_daily_multi(symbols, board, start, end)
    n_sym = len(sorted(df["ticker"].unique()))
    env = MultiTickerEnv(df, window=window, feature_cols=feature_cols)
    model = PPO.load(model_path)

    try:
        expected_shape = model.observation_space.shape
    except Exception:
        expected_shape = model.policy.observation_space.shape

    actual_shape = env.observation_space.shape
    print(f"Model expected observation shape: {expected_shape}")
    print(f"Env actual observation shape:     {actual_shape}")
    exp_dim = expected_shape[0]
    act_dim = actual_shape[0]
    base = 1 + n_sym
    exp_window_plus_feats = (exp_dim - base) / n_sym
    act_window_plus_feats = (act_dim - base) / n_sym
    print(f"n_sym={n_sym}, base(1+n_sym)={base}")
    print(f"model (window + n_feat) = {exp_window_plus_feats}")
    print(f"env   (window + n_feat) = {act_window_plus_feats}")
    if exp_dim != act_dim:
        print("\nERROR: observation shape mismatch.")
        print(
            "Make sure to create the environment with the SAME window and feature columns"
        )
        print("that were used during training.")
        print("\nSuggested run (use these values):")

        inferred_total = int(exp_window_plus_feats)
        print(f"  window + n_feature_cols should be = {inferred_total}")
        print(
            "If training used window=20, then n_feature_cols = "
            f"{inferred_total - 20} (adjust if negative)."
        )
        raise SystemExit(1)

    # run evaluation
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
    parser.add_argument("--board", default="TQBR")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-06-01")
    parser.add_argument(
        "--window", type=int, default=20, help="window length used during training"
    )
    parser.add_argument(
        "--feature-cols",
        nargs="*",
        default=[
            "sma20",
            "boll_upper20",
            "boll_lower20",
            "macd",
            "macd_signal",
            "rsi14",
        ],
        help="feature columns used during training (space-separated)",
    )
    args = parser.parse_args()

    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)

    env, hist = evaluate(
        args.model, args.symbols, args.board, start, end, args.window, args.feature_cols
    )

    # plotting (optional)
    import matplotlib.pyplot as plt

    plt.plot(hist)
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Step")
    plt.ylabel("Portfolio Value")
    plt.show()
