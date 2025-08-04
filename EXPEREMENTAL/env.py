from collections import deque

import numpy as np
from gym import spaces
from sklearn.preprocessing import StandardScaler


class MultiTimeframeEnv:
    def __init__(self, df, minute_window=60, daily_window=5, init_balance=1e6):
        self.df = df.sort_index()
        self.tickers = df["ticker"].unique().tolist()
        self.n_tickers = len(self.tickers)
        self.minute_window = minute_window
        self.daily_window = daily_window
        self.init_balance = init_balance

        self.minute_buffer = {t: deque(maxlen=minute_window) for t in self.tickers}
        self.daily_buffer = {t: deque(maxlen=daily_window) for t in self.tickers}

        self.scalers = {}
        self._init_scalers()

        # Определение пространства действий и состояний
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_tickers,))
        self.observation_space = self._get_observation_space()

        self.reset()

    def _init_scalers(self):
        features = [
            "close_min",
            "volume_min",
            "sma20_min",
            "rsi_min",
            "close_eod",
            "volume_eod",
            "sma5_eod",
            "daily_volatility",
            "premium_to_daily",
        ]

        for feature in features:
            if feature in self.df.columns:
                scaler = StandardScaler()
                scaler.fit(self.df[feature].values.reshape(-1, 1))
                self.scalers[feature] = scaler

    def _get_observation_space(self):
        # Размер состояния
        minute_feats = 5  # close, volume, sma20, rsi, premium
        daily_feats = 4  # close, sma5, volatility, volume

        obs_size = (
            minute_feats * self.n_tickers * self.minute_window
            + daily_feats * self.n_tickers * self.daily_window
            + 1
            + self.n_tickers
        )

        return spaces.Box(low=-5, high=5, shape=(obs_size,))

    def reset(self):
        self.current_step = 0
        self.cash = self.init_balance
        self.positions = {t: 0 for t in self.tickers}
        self.portfolio_value = [self.init_balance]
        self.peak_value = self.init_balance

        # Инициализация буферов начальными данными
        for t in self.tickers:
            ticker_data = self.df[self.df["ticker"] == t]
            for i in range(self.minute_window):
                self.minute_buffer[t].append(ticker_data.iloc[i])
            for i in range(self.daily_window):
                self.daily_buffer[t].append(ticker_data.iloc[i])

        return self._get_state()

    def _get_state(self):
        state = []

        # Минутные данные
        for t in self.tickers:
            for data_point in self.minute_buffer[t]:
                normalized = [
                    self.scalers["close_min"].transform([[data_point["close_min"]]])[0][
                        0
                    ],
                    self.scalers["volume_min"].transform([[data_point["volume_min"]]])[
                        0
                    ][0],
                    self.scalers["sma20_min"].transform([[data_point["sma20_min"]]])[0][
                        0
                    ],
                    self.scalers["rsi_min"].transform([[data_point["rsi_min"]]])[0][0],
                    self.scalers["premium_to_daily"].transform(
                        [[data_point["premium_to_daily"]]]
                    )[0][0],
                ]
                state.extend(normalized)

        # Дневные данные
        for t in self.tickers:
            for data_point in self.daily_buffer[t]:
                normalized = [
                    self.scalers["close_eod"].transform([[data_point["close_eod"]]])[0][
                        0
                    ],
                    self.scalers["sma5_eod"].transform([[data_point["sma5_eod"]]])[0][
                        0
                    ],
                    self.scalers["daily_volatility"].transform(
                        [[data_point["daily_volatility"]]]
                    )[0][0],
                    self.scalers["volume_eod"].transform([[data_point["volume_eod"]]])[
                        0
                    ][0],
                ]
                state.extend(normalized)

        # Финансовые показатели
        state.append(self.cash / self.init_balance)
        for t in self.tickers:
            position_value = self.positions[t] * self._current_price(t)
            state.append(position_value / self.init_balance)

        return np.array(state, dtype=np.float32)

    def step(self, actions):
        # Выполнение действий
        self._execute_actions(actions)

        # Переход к следующему шагу
        self.current_step += 1
        if self.current_step >= len(self.df):
            done = True
            next_state = np.zeros(self.observation_space.shape)
        else:
            self._update_buffers()
            next_state = self._get_state()
            done = False

        # Расчет вознаграждения
        reward = self._calculate_reward()

        # Обновление истории портфеля
        current_value = self._portfolio_value()
        self.portfolio_value.append(current_value)
        self.peak_value = max(self.peak_value, current_value)

        return next_state, reward, done, {"portfolio_value": current_value}

    def _execute_actions(self, actions):
        total_value = self._portfolio_value()

        for i, t in enumerate(self.tickers):
            target_allocation = actions[i]
            current_price = self._current_price(t)
            current_value = self.positions[t] * current_price
            target_value = total_value * target_allocation

            # Рассчет необходимой корректировки
            delta_value = target_value - current_value

            if delta_value > 0:  # Покупка
                shares_to_buy = delta_value / current_price
                self.cash -= delta_value
                self.positions[t] += shares_to_buy
            else:  # Продажа
                shares_to_sell = -delta_value / current_price
                if shares_to_sell <= self.positions[t]:
                    self.cash += -delta_value
                    self.positions[t] -= shares_to_sell

    def _calculate_reward(self):
        current_value = self._portfolio_value()
        prev_value = self.portfolio_value[-1]

        returns = (current_value - prev_value) / prev_value if prev_value > 0 else 0

        # Штраф за просадку
        drawdown = (self.peak_value - current_value) / self.peak_value
        drawdown_penalty = 0.5 * drawdown

        # Штраф за волатильность
        volatility_penalty = (
            0.1 * np.std(self.portfolio_value[-10:])
            if len(self.portfolio_value) > 10
            else 0
        )

        return returns - drawdown_penalty - volatility_penalty

    def _current_price(self, ticker):
        # Получение текущей цены для тикера
        ticker_data = self.df[self.df["ticker"] == ticker]
        if self.current_step < len(ticker_data):
            return ticker_data.iloc[self.current_step]["close_min"]
        return ticker_data.iloc[-1]["close_min"]

    def _portfolio_value(self):
        value = self.cash
        for t in self.tickers:
            value += self.positions[t] * self._current_price(t)
        return value

    def _update_buffers(self):
        for t in self.tickers:
            ticker_data = self.df[self.df["ticker"] == t]
            if self.current_step < len(ticker_data):
                self.minute_buffer[t].append(ticker_data.iloc[self.current_step])

                # Обновление дневных данных при смене дня
                current_date = ticker_data.iloc[self.current_step]["date"]
                if (
                    len(self.daily_buffer[t]) == 0
                    or current_date != self.daily_buffer[t][-1]["date"]
                ):
                    if len(ticker_data) > self.current_step:
                        self.daily_buffer[t].append(ticker_data.iloc[self.current_step])
