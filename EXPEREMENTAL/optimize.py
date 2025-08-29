import optuna
from data import main as load_data
from env import MultiTimeframeEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


async def optimize_hyperparameters(tickers, date_from, date_to):
    df = await load_data(tickers, date_from, date_to)

    def objective(trial):
        env = MultiTimeframeEnv(df)
        vec_env = DummyVecEnv([lambda: env])

        # Гиперпараметры для оптимизации
        lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
        n_steps = trial.suggest_categorical("n_steps", [256, 512, 1024, 2048])
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        gamma = trial.suggest_uniform("gamma", 0.9, 0.9999)
        gae_lambda = trial.suggest_uniform("gae_lambda", 0.8, 0.999)
        clip_range = trial.suggest_uniform("clip_range", 0.1, 0.4)
        ent_coef = trial.suggest_loguniform("ent_coef", 0.0001, 0.1)

        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=lr,
            n_steps=n_steps,
            batch_size=batch_size,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            verbose=0,
            tensorboard_log="./tensorboard/",
        )

        model.learn(total_timesteps=100000)

        # Оценка производительности
        total_reward = 0
        obs = vec_env.reset()
        for _ in range(1000):
            action, _ = model.predict(obs)
            obs, reward, done, _ = vec_env.step(action)
            total_reward += reward[0]
            if done:
                break

        return total_reward

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    return study.best_params
