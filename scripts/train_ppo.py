import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import gymnasium as gym
from datetime import datetime
import numpy as np
import yaml
import json
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from moex_rl_agent.data_loader import load_daily_multi
from moex_rl_agent.env import MultiTickerEnv


def load_config(config_path: str) -> dict:
    """Загрузка конфигурации"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"❌ Config file not found: {path.resolve()}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f) or {}
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        raise


def create_env(
    symbols: list,
    board: str,
    start: datetime,
    end: datetime,
    window: int,
    feature_cols: list,
    env_kwargs: dict,
    seed: int = 42,
    monitor_file: str = None
):
    """Создание среды с валидацией данных"""
    try:
        print(f"\n Loading data for {len(symbols)} symbols...")
        print(f"   Period: {start.date()} → {end.date()}")
        
        df = load_daily_multi(symbols, board, start, end, use_cache=True)
        
        if df.empty:
            raise ValueError("❌ No data loaded!")
        
        # Проверяем фичи
        available_features = [c for c in df.columns if c not in ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']]
        missing_features = [f for f in feature_cols if f not in df.columns]
        if missing_features:
            print(f"   ⚠️  Missing features: {missing_features}")
            print(f"   Available: {available_features}")
            feature_cols = [f for f in feature_cols if f in df.columns]
        
        print(f"✅ Data loaded: {df.shape[0]} rows, {len(df['ticker'].unique())} tickers")
        print(f"   Features: {feature_cols}")
        
        env = MultiTickerEnv(
            df,
            window=window,
            feature_cols=feature_cols,
            seed=seed,
            **env_kwargs
        )
        
        if monitor_file:
            env = Monitor(env, monitor_file)
        
        return env
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Train MOEX RL Agent")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--symbols", nargs="+", help="Override symbols")
    parser.add_argument("--total-timesteps", type=int, help="Override timesteps")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    
    args = parser.parse_args()
    
    # Загрузка конфига
    print(f"Loading config from {args.config}...")
    config = load_config(args.config)
    
    # Переопределение
    if args.symbols:
        config["symbols"] = args.symbols
    if args.total_timesteps:
        config["total_timesteps"] = args.total_timesteps
    
    # Параметры
    symbols = config.get("symbols", ["SBER", "GAZP", "LKOH"])
    board = config.get("board", "TQBR")
    start = datetime.fromisoformat(config["start"])
    end = datetime.fromisoformat(config["end"])
    val_start = datetime.fromisoformat(config.get("val_start", "2024-01-01"))
    val_end = datetime.fromisoformat(config.get("val_end", "2025-01-01"))
    window = config.get("window", 20)
    feature_cols = config.get("feature_cols", ["sma20", "rsi14"])
    total_timesteps = config.get("total_timesteps", 200_000)
    seed = config.get("seed", 42)
    model_path = config.get("model_path", "models/ppo_moex")
    env_kwargs = config.get("env_kwargs", {})
    tensorboard_enabled = config.get("tensorboard", False)
    progress_bar_enabled = config.get("progress_bar", True) and not args.no_progress
    
    # Вывод конфигурации
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Symbols:   {symbols}")
    print(f"Board:     {board}")
    print(f"Period:    {start.date()} → {end.date()}")
    print(f"Window:    {window}")
    print(f"Features:  {feature_cols}")
    print(f"Timesteps: {total_timesteps:,}")
    print(f"Device:    {'CPU' if args.cpu else 'auto'}")
    print(f"Progress:  {'ON' if progress_bar_enabled else 'OFF'}")
    print(f"Tensorboard: {'ON' if tensorboard_enabled else 'OFF'}")
    print("="*60 + "\n")
    
    # Создание директории
    Path(model_path).parent.mkdir(exist_ok=True)
    
    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=str(Path(model_path).parent),
        name_prefix=Path(model_path).stem,
        save_replay_buffer=False
    )
    
    # Среды
    print("[1/3] Creating training environment...")
    train_env = DummyVecEnv([lambda: create_env(
        symbols, board, start, end, window, feature_cols, env_kwargs, seed,
        monitor_file=str(Path(model_path).parent / "train_monitor.csv")
    )])
    
    print("[2/3] Creating validation environment...")
    val_env = DummyVecEnv([lambda: create_env(
        symbols, board, val_start, val_end, window, feature_cols, env_kwargs, seed + 1,
        monitor_file=str(Path(model_path).parent / "val_monitor.csv")
    )])
    
    eval_cb = EvalCallback(
        val_env,
        best_model_save_path=str(Path(model_path).parent),
        log_path=str(Path(model_path).parent),
        eval_freq=10_000,
        n_eval_episodes=3,
        deterministic=True,
        verbose=1
    )
    
    # Модель
    print("[3/3] Creating PPO model...")
    device = 'cpu' if args.cpu else 'auto'
    
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        n_epochs=10,
        ent_coef=0.0,
        device=device,
        tensorboard_log=str(Path(model_path).parent / "tb_logs") if tensorboard_enabled else None
    )
    
    # Обучение
    print(f"\n Starting training for {total_timesteps:,} timesteps...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_cb, eval_cb],
            progress_bar=progress_bar_enabled
        )
        
        final_path = f"{model_path}_final.zip"
        model.save(final_path)
        print(f"\n✅ Training complete! Final model: {final_path}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
        interrupted_path = f"{model_path}_interrupted.zip"
        model.save(interrupted_path)
        print(f"💾 Partially trained model saved: {interrupted_path}")
    
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()