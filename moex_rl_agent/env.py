from typing import List, Optional

import gym
import numpy as np
import pandas as pd
from gym import spaces
from sklearn.preprocessing import StandardScaler


class UnifiedBuffer:
    def __init__(self, maxlen: int = 1000):
        self.maxlen = maxlen
        self.buf = []

    def update(self, x):
        self.buf.append(x)
        if len(self.buf) > self.maxlen:
            self.buf.pop(0)

    def get(self):
        return np.array(self.buf)


class MultiTickerEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        window: int = 20,
        init_balance: float = 1e6,
        commission: float = 0.0005,
        slippage: float = 0.0002,
        dd_penalty: float = 0.3,
        normalize_action: bool = True,
        max_weight_per_asset: float = 0.8,
        cooldown_steps: int = 1,
        min_trade_value: float = 100.0,
        max_trade_pct: float = 0.2,
        turnover_penalty: float = 0.0,
        weight_smoothing_alpha: Optional[float] = None,
        feature_cols: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        assert {"date", "ticker", "close"}.issubset(
            set(df.columns)
        ), "df must contain date,ticker,close"
        self.raw = df.copy()
        self.window = window
        self.init_balance = init_balance
        self.comm = commission
        self.slip = slippage
        self.dd_penalty = dd_penalty
        self.normalize_action = normalize_action

        self.max_weight_per_asset = max_weight_per_asset
        self.cooldown_steps = int(cooldown_steps)
        self.min_trade_value = float(min_trade_value)
        self.max_trade_pct = float(max_trade_pct)
        self.turnover_penalty = float(turnover_penalty)
        self.weight_smoothing_alpha = weight_smoothing_alpha

        self.feature_cols = feature_cols or []

        pivot = self.raw.pivot(
            index="date", columns="ticker", values="close"
        ).sort_index()
        self.dates = list(pivot.index)
        self.symbols = list(pivot.columns)
        self.n_sym = len(self.symbols)
        self.prices = pivot.values.astype(float)  # (T, n_sym)

        self.scalers = {}
        norm_prices = np.zeros_like(self.prices)
        for i, sym in enumerate(self.symbols):
            sc = StandardScaler()
            norm_prices[:, i] = sc.fit_transform(
                self.prices[:, i].reshape(-1, 1)
            ).flatten()
            self.scalers[sym] = sc
        self.norm_prices = norm_prices

        self.feature_pivots = {}
        self.feature_scalers = {}
        for f in self.feature_cols:
            if f in self.raw.columns:
                pivot_f = self.raw.pivot(
                    index="date", columns="ticker", values=f
                ).sort_index()
                arr = pivot_f.values.astype(float)
                scaled = np.zeros_like(arr)
                scalers = []
                for i in range(arr.shape[1]):
                    sc = StandardScaler()
                    try:
                        scaled[:, i] = sc.fit_transform(
                            arr[:, i].reshape(-1, 1)
                        ).flatten()
                    except Exception:
                        scaled[:, i] = arr[:, i]  # fallback
                    scalers.append(sc)
                self.feature_pivots[f] = scaled
                self.feature_scalers[f] = scalers

        # action & observation spaces
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_sym,), dtype=np.float32
        )
        obs_dim = (
            self.window * self.n_sym
            + len(self.feature_cols) * self.n_sym
            + 1
            + self.n_sym
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # state
        self.buffer = UnifiedBuffer(maxlen=1000)
        self.last_trade_step = np.full(self.n_sym, -9999, dtype=int)
        self.weight_smoothed = np.zeros(self.n_sym, dtype=float)

        self._reset_internal()
        self.seed(seed)

    def _reset_internal(self):
        self.t = self.window
        self.cash = self.init_balance
        self.shares = np.zeros(self.n_sym, dtype=float)
        self.weights = np.zeros(self.n_sym, dtype=float)
        self.port_hist = [self.init_balance]
        self.peak = self.init_balance

    def seed(self, seed=None):
        np.random.seed(seed)

    def reset(self):
        self._reset_internal()
        self.last_trade_step = np.full(self.n_sym, -9999, dtype=int)
        self.weight_smoothed = np.zeros(self.n_sym, dtype=float)
        return self._get_obs()

    def _get_obs(self):
        block = self.norm_prices[self.t - self.window : self.t].flatten()
        feat_block = []
        for f in self.feature_cols:
            arr = self.feature_pivots.get(f)
            if arr is None:
                feat_block.extend([0.0] * self.n_sym)
            else:
                feat_block.extend(arr[self.t].tolist())  # current step features
        cash_frac = np.array([self.cash / (self.init_balance + 1e-9)])
        obs = np.concatenate(
            [block, np.array(feat_block), cash_frac, self.weights]
        ).astype(np.float32)
        return obs

    def step(self, action):
        a = np.clip(action, 0.0, 1.0).astype(float)
        a = np.minimum(a, self.max_weight_per_asset)
        if self.normalize_action:
            s = a.sum()
            if s > 0:
                a = a / s

        cooldown_mask = (self.t - self.last_trade_step) < self.cooldown_steps
        if cooldown_mask.any():
            a[cooldown_mask] = self.weights[cooldown_mask]

        if self.weight_smoothing_alpha is not None:
            alpha = float(self.weight_smoothing_alpha)
            a = alpha * a + (1.0 - alpha) * self.weight_smoothed
            if self.normalize_action:
                s = a.sum()
                if s > 0:
                    a = a / s

        prices = self.prices[self.t]
        total_val = self.cash + (self.shares * prices).sum()
        target_vals = total_val * a
        curr_vals = self.shares * prices
        deltas = target_vals - curr_vals

        max_trade_val = self.max_trade_pct * total_val
        deltas = np.sign(deltas) * np.minimum(np.abs(deltas), max_trade_val)
        small_trade_mask = np.abs(deltas) < self.min_trade_value
        deltas[small_trade_mask] = 0.0

        trade_vals = np.abs(deltas)
        comm_costs = trade_vals * self.comm
        slip_costs = trade_vals * self.slip

        executed_vals = np.zeros_like(deltas)

        sell_mask = deltas < 0
        if sell_mask.any():
            qty_sell = -deltas[sell_mask] / prices[sell_mask]
            proceeds = (
                qty_sell * prices[sell_mask]
                - comm_costs[sell_mask]
                - slip_costs[sell_mask]
            )
            self.shares[sell_mask] -= qty_sell
            executed_vals[sell_mask] = np.abs(deltas[sell_mask])
            self.cash += proceeds.sum()

        buy_mask = deltas > 0
        if buy_mask.any():
            desired_costs = (
                deltas[buy_mask] + comm_costs[buy_mask] + slip_costs[buy_mask]
            )
            total_desired = desired_costs.sum()
            if total_desired <= self.cash + 1e-9:
                qty_buy = deltas[buy_mask] / prices[buy_mask]
                self.shares[buy_mask] += qty_buy
                executed_vals[buy_mask] = deltas[buy_mask]
                self.cash -= total_desired
            else:
                scale = (self.cash) / (total_desired + 1e-12)
                scaled_deltas = deltas[buy_mask] * scale
                scaled_comm = comm_costs[buy_mask] * scale
                scaled_slip = slip_costs[buy_mask] * scale
                qty_buy = scaled_deltas / prices[buy_mask]
                self.shares[buy_mask] += qty_buy
                executed_vals[buy_mask] = scaled_deltas
                spent = (scaled_deltas + scaled_comm + scaled_slip).sum()
                self.cash -= spent

        self.shares = np.maximum(self.shares, 0.0)
        port_val = self.cash + (self.shares * prices).sum()
        self.weights = ((self.shares * prices) / (port_val + 1e-9)).astype(float)
        self.weight_smoothed = self.weights.copy()

        traded_mask = executed_vals > 0.0
        if traded_mask.any():
            self.last_trade_step[traded_mask] = self.t

        self.t += 1
        done = self.t >= len(self.dates)

        self.port_hist.append(port_val)
        self.peak = max(self.peak, port_val)
        drawdown = (self.peak - port_val) / (self.peak + 1e-9)
        pnl = (self.port_hist[-1] - self.port_hist[-2]) / (self.port_hist[-2] + 1e-9)
        turnover = executed_vals.sum() / (total_val + 1e-9)

        reward = pnl - self.dd_penalty * drawdown - self.turnover_penalty * turnover

        obs = (
            self._get_obs()
            if not done
            else np.zeros(self.observation_space.shape, dtype=np.float32)
        )
        info = {
            "portfolio_value": float(port_val),
            "cash": float(self.cash),
            "turnover": float(turnover),
        }
        return obs, float(reward), bool(done), info

    def render(self, mode="human"):
        date = self.dates[self.t - 1]
        print(
            f"{date}: PV={self.port_hist[-1]:.2f}, cash={self.cash:.2f}, weights={self.weights}"
        )
