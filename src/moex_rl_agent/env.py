import gym
import numpy as np
from gym import spaces
from sklearn.preprocessing import StandardScaler


class MultiTickerEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        window: int = 20,
        init_balance: float = 1e6,
        commission: float = 0.0005,
        slippage: float = 0.0002,
        dd_penalty: float = 0.3,
    ):
        super().__init__()
        self.raw = df.copy()
        self.symbols = sorted(df["ticker"].unique())
        self.n_sym = len(self.symbols)
        self.window = window
        self.init_balance = init_balance
        self.comm = commission
        self.slip = slippage
        self.dd_penalty = dd_penalty

        # Матрица цен
        pivot = df.pivot(index="date", columns="ticker", values="close")
        self.dates = pivot.index.to_list()
        prices = pivot[self.symbols].values  # shape (T, n_sym)

        # Нормализуем отдельно каждый тикер
        norm_prices = np.zeros_like(prices)
        self.scalers = {}
        for i, sym in enumerate(self.symbols):
            sc = StandardScaler().fit(prices[:, i].reshape(-1, 1))
            norm_prices[:, i] = sc.transform(prices[:, i].reshape(-1, 1)).flatten()
            self.scalers[sym] = sc
        self.norm_prices = norm_prices
        self.prices = prices

        # Action: continuous vector in [0,1]^n_sym
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_sym,), dtype=np.float32
        )

        # Observation: window × n_sym norm_prices + cash_ratio + weights
        obs_dim = self.window * self.n_sym + 1 + self.n_sym
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._reset_internal()

    def _reset_internal(self):
        self.t = self.window
        self.cash = self.init_balance
        self.shares = np.zeros(self.n_sym, dtype=np.float32)
        self.weights = np.zeros(self.n_sym, dtype=np.float32)
        self.port_hist = [self.init_balance]
        self.peak = self.init_balance

    def reset(self):
        self._reset_internal()
        return self._get_obs()

    def _get_obs(self):
        block = self.norm_prices[self.t - self.window : self.t].flatten()
        return np.concatenate(
            [block, [self.cash / self.init_balance], self.weights]
        ).astype(np.float32)

    def step(self, action):
        a = np.clip(action, 0, 1)
        prices = self.prices[self.t]
        total_val = self.cash + np.dot(self.shares, prices)

        target_vals = total_val * a
        curr_vals = self.shares * prices
        deltas = target_vals - curr_vals

        # комиссии и проскальзывание
        comm_cost = np.abs(deltas) * self.comm
        slip_cost = np.abs(deltas) * self.slip

        # BUY
        buy = deltas > 0
        cost = deltas[buy] + comm_cost[buy] + slip_cost[buy]
        if self.cash >= cost.sum():
            self.cash -= cost.sum()
            self.shares[buy] += deltas[buy] / prices[buy]

        # SELL
        sell = deltas < 0
        proceeds = -deltas[sell] - comm_cost[sell] - slip_cost[sell]
        self.cash += proceeds.sum()
        self.shares[sell] += deltas[sell] / prices[sell]

        self.shares = np.maximum(self.shares, 0.0)
        self.weights = a
        self.t += 1
        done = self.t >= len(self.dates)

        port = self.cash + np.dot(self.shares, prices)
        self.port_hist.append(port)
        self.peak = max(self.peak, port)
        drawdown = (self.peak - port) / self.peak

        pnl = (self.port_hist[-1] - self.port_hist[-2]) / (self.port_hist[-2] + 1e-9)
        reward = pnl - self.dd_penalty * drawdown

        obs = (
            self._get_obs()
            if not done
            else np.zeros(self.observation_space.shape, dtype=np.float32)
        )
        return obs, reward, done, {"portfolio_value": port}

    def render(self, mode="human"):
        date = self.dates[self.t - 1]
        print(f"{date.date()}: PV={self.port_hist[-1]:.2f}, weights={self.weights}")
