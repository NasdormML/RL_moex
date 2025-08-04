from data import main as load_data
from env import MultiTimeframeEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback


async def train_model(tickers, date_from, date_to):
    df = await load_data(tickers, date_from, date_to)

    env = MultiTimeframeEnv(df)

    checkpoint_callback = CheckpointCallback(
        save_freq=100000, save_path="./checkpoints/", name_prefix="rl_model"
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./tensorboard/",
        device="auto",
        # Здесь должны быть лучшие параметры из optimize_ppo
    )

    model.learn(
        total_timesteps=1_000_000,
        callback=checkpoint_callback,
        tb_log_name="ppo_training",
    )

    # Сохранение модели
    model.save("ppo_multi_ticker")
    return model
