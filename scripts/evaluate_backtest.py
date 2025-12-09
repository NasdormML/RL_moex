import argparse
from datetime import datetime
import json
from typing import Any, Dict, List, Optional

import numpy as np
from stable_baselines3 import PPO

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from moex_rl_agent.data_loader import load_daily_multi
from moex_rl_agent.env import MultiTickerEnv


def verify_observation_space(
    model: PPO,
    env: MultiTickerEnv,
    n_sym: int
) -> bool:
    """Проверка совместимости размерностей"""
    try:
        expected_shape = model.observation_space.shape
    except Exception:
        expected_shape = model.policy.observation_space.shape
    
    actual_shape = env.observation_space.shape
    
    print(f"Model expected: {expected_shape}")
    print(f"Env actual:     {actual_shape}")
    
    if expected_shape != actual_shape:
        exp_dim, act_dim = expected_shape[0], actual_shape[0]
        base = 1 + n_sym
        exp_window_plus_feats = (exp_dim - base) / n_sym
        act_window_plus_feats = (act_dim - base) / n_sym
        
        print(f"n_sym={n_sym}, base(1+n_sym)={base}")
        print(f"Model (window + n_feat) = {exp_window_plus_feats}")
        print(f"Env (window + n_feat) = {act_window_plus_feats}")
        
        print("\nERROR: Observation shape mismatch!")
        print("Use same window and feature_cols as in training")
        raise SystemExit(1)
    
    return True


def run_backtest(
    model_path: str,
    symbols: List[str],
    board: str,
    start: datetime,
    end: datetime,
    window: int,
    feature_cols: List[str],
    **env_kwargs
) -> Dict[str, Any]:
    """Запуск полного бэктеста"""
    print("Loading data...")
    df = load_daily_multi(symbols, board, start, end, use_cache=True)
    
    if df.empty:
        raise ValueError("No data loaded")
    
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    print("\nCreating environment...")
    env = MultiTickerEnv(
        df,
        window=window,
        feature_cols=feature_cols,
        **env_kwargs
    )
    
    print("\nLoading model...")
    model = PPO.load(model_path)
    
    # Проверка размерности
    verify_observation_space(model, env, len(symbols))
    
    print("\nRunning backtest...")
    obs = env.reset()
    done = False
    
    # Для сбора статистики
    portfolio_values = []
    weights_history = []
    actions_history = []
    dates = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        
        obs, reward, done, info = env.step(action)
        
        # Сохранение истории
        portfolio_values.append(info["portfolio_value"])
        weights_history.append(info["weights"])
        actions_history.append(action)
        dates.append(env.dates[env.current_step - 1])
    
    print("\nBacktest complete!")
    
    # Финальные метрики
    metrics = info.get("final_metrics", {})
    
    return {
        "env": env,
        "dates": dates,
        "portfolio_values": np.array(portfolio_values),
        "weights_history": np.array(weights_history),
        "actions_history": np.array(actions_history),
        "metrics": metrics,
    }


def print_metrics(metrics: Dict[str, float]):
    print("\n" + "="*50)
    print("FINAL METRICS")
    print("="*50)
    print(f"Total Return:  {metrics['total_return']*100:>8.2f}%")
    print(f"Final Value:   {metrics['final_value']:>8,.0f}")
    print(f"Sharpe Ratio:  {metrics['sharpe_ratio']:>8.2f}")
    print(f"Max Drawdown:  {metrics['max_drawdown']*100:>8.2f}%")
    print(f"Calmar Ratio:  {metrics['calmar_ratio']:>8.2f}")
    print(f"Total Trades:  {metrics['total_trades']:>8,d}")
    print("="*50)


def plot_results(results: Dict[str, Any], save_path: Optional[str] = None):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    dates = results["dates"]
    values = results["portfolio_values"]
    weights = results["weights_history"]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])
    
    # График стоимости портфеля
    ax1.plot(dates, values, label="Portfolio", linewidth=2)
    ax1.set_title("Portfolio Value Over Time", fontsize=16)
    ax1.set_ylabel("Value (₽)", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Форматирование дат
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    
    # График весов
    for i, symbol in enumerate(results["env"].symbols):
        ax2.plot(dates, weights[:, i], label=symbol, alpha=0.7)
    
    ax2.set_ylabel("Weights", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MOEX RL Agent")
    parser.add_argument("--model", required=True, help="Path to saved model")
    parser.add_argument("--symbols", nargs="+", default=["SBER", "GAZP", "LKOH"])
    parser.add_argument("--board", default="TQBR")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-09-12")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--feature-cols", nargs="*", default=[
        "sma20", "boll_upper20", "boll_lower20", "macd", "macd_signal", "rsi14"
    ])
    parser.add_argument("--save-plot", help="Save plot to file")
    parser.add_argument("--env-kwargs", type=json.loads, default="{}")
    
    args = parser.parse_args()
    
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    
    results = run_backtest(
        args.model,
        args.symbols,
        args.board,
        start,
        end,
        args.window,
        args.feature_cols,
        **args.env_kwargs
    )
    
    print_metrics(results["metrics"])
    
    if args.save_plot or True:
        plot_results(results, args.save_plot)