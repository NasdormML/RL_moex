from typing import List, Optional, Dict
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", module="sklearn")


class UnifiedBuffer:
    """Кольцевой буфер для хранения истории доходностей"""
    def __init__(self, maxlen: int = 1000):
        self.maxlen = maxlen
        self.buf = []
    
    def append(self, x):
        self.buf.append(x)
        if len(self.buf) > self.maxlen:
            self.buf.pop(0)
    
    def get(self, n: Optional[int] = None) -> np.ndarray:
        """Получить последние n значений"""
        data = self.buf[-n:] if n else self.buf
        return np.array(data)
    
    def clear(self):
        self.buf.clear()


class MultiTickerEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        df: pd.DataFrame,
        window: int = 20,
        init_balance: float = 1_000_000.0,
        commission: float = 0.0005,  # 0.05% комиссия
        slippage: float = 0.0002,    # 0.02% проскальзывание
        dd_penalty: float = 0.3,     # Пенальти за просадку
        normalize_action: bool = True,
        max_weight_per_asset: float = 0.8,
        cooldown_steps: int = 1,     # Шагов между сделками
        min_trade_value: float = 100.0,  # Минимальная сделка
        max_trade_pct: float = 0.2,  # Макс % от портфеля за шаг
        turnover_penalty: float = 0.0,
        weight_smoothing_alpha: Optional[float] = None,
        feature_cols: Optional[List[str]] = None,
        seed: Optional[int] = None,
        allow_short: bool = False,  # Разрешить шорт
        risk_free_rate: float = 0.0,
        fill_nan_prices: bool = True,
    ):
        super().__init__()
        
        # Валидация входных данных
        required_cols = {"date", "ticker", "close"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"❌ df must contain {required_cols}, found {df.columns.tolist()}")
        
        if allow_short:
            raise NotImplementedError("❌ Short selling is not properly implemented yet")
        
        self.raw = df.copy()
        self.window = window
        self.init_balance = init_balance
        self.comm = commission
        self.slip = slippage
        self.dd_penalty = dd_penalty
        self.normalize_action = normalize_action
        self.allow_short = allow_short
        self.risk_free_rate = risk_free_rate
        self.fill_nan_prices = fill_nan_prices
        
        # Параметры торговли
        self.max_weight_per_asset = max_weight_per_asset
        self.cooldown_steps = int(cooldown_steps)
        self.min_trade_value = float(min_trade_value)
        self.max_trade_pct = float(max_trade_pct)
        self.turnover_penalty = float(turnover_penalty)
        self.weight_smoothing_alpha = weight_smoothing_alpha
        
        self.feature_cols = feature_cols or []
        
        # Подготовка данных: пивоты, НЕ НОРМАЛИЗУЕМ ЦЕНЫ (УБРАНО)
        self._prepare_data()
        # Расчет размерности observation
        self._calculate_obs_dim()
        # Настройка пространств действий и наблюдений
        self._setup_spaces()
        # Инициализация состояния
        self._reset_internal()
        
        if seed is not None:
            self.seed(seed)
    
    def _prepare_data(self):
        # === ОБРАБОТКА ЦЕН ===
        pivot = self.raw.pivot(
            index="date", columns="ticker", values="close"
        ).sort_index()
        
        self.dates = list(pivot.index)
        self.symbols = list(pivot.columns)
        self.n_sym = len(self.symbols)
        self.prices = pivot.values.astype(np.float64)
        
        if np.any(np.isnan(self.prices)):
            print(f"⚠️  NaN found in price data for {self.symbols}")
            
            if self.fill_nan_prices:
                print("   Applying forward-fill interpolation...")
                
                # Интерполяция для каждого тикера
                for i in range(self.prices.shape[1]):
                    col = self.prices[:, i]
                    
                    # Ищем первое валидное значение
                    first_valid_idx = np.where(~np.isnan(col))[0]
                    if len(first_valid_idx) == 0:
                        print(f"      All NaN for {self.symbols[i]}, filling with 0")
                        col[:] = 0.0
                        continue
                    
                    # Forward-fill до первого валидного значения
                    if first_valid_idx[0] > 0:
                        col[:first_valid_idx[0]] = col[first_valid_idx[0]]
                    
                    # Интерполяция для остальных пропусков
                    valid_mask = ~np.isnan(col)
                    if not np.all(valid_mask):
                        indices = np.arange(len(col))
                        col[~valid_mask] = np.interp(
                            indices[~valid_mask],
                            indices[valid_mask],
                            col[valid_mask]
                        )
                
                if np.any(np.isnan(self.prices)):
                    raise ValueError("❌ NaN persist after filling!")
                
                print("   ✅ Price NaN resolved")
            else:
                raise ValueError("❌ NaN values found in price data! Set fill_nan_prices=True")
        
        # УБРАНА НОРМАЛИЗАЦИЯ ЦЕН (look-ahead bias)
        self.norm_prices = self.prices.copy()  # Используем необработанные цены
        
        # === ОБРАБОТКА ФИЧЕЙ ===
        self.feature_data = {}
        self.feature_scalers = {}
        
        # УБРАНА НОРМАЛИЗАЦИЯ ФИЧЕЙ (look-ahead bias)
        for f in self.feature_cols:
            if f not in self.raw.columns:
                print(f"⚠️  Feature '{f}' not found in data, using zeros")
                self.feature_data[f] = np.zeros((len(self.dates), self.n_sym))
                continue
            
            pivot_f = self.raw.pivot(
                index="date", columns="ticker", values=f
            ).sort_index().values.astype(np.float64)
            
            # Обработка NaN в фичах
            if np.any(np.isnan(pivot_f)):
                print(f"⚠️  NaN in feature '{f}', filling with 0")
                pivot_f = np.nan_to_num(pivot_f, nan=0.0, posinf=0.0, neginf=0.0)
            
            # УБРАНА НОРМАЛИЗАЦИЯ
            self.feature_data[f] = pivot_f
            
            # Финальная проверка на NaN
            if np.any(np.isnan(self.feature_data[f])):
                raise ValueError(f"❌ NaN in feature {f} after processing!")
    
    def _calculate_obs_dim(self):
        """Расчет размерности observation"""
        self.obs_dim = (
            self.window * self.n_sym +
            len(self.feature_cols) * self.n_sym +
            1 +
            self.n_sym
        )
    
    def _setup_spaces(self):
        # Action space: [0, 1] для лонга (шорт НЕ ПОДДЕРЖИВАЕТСЯ)
        low = 0.0
        self.action_space = spaces.Box(
            low=low, high=1.0, shape=(self.n_sym,), dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
    
    def _reset_internal(self):
        """Внутренний сброс состояния"""
        self.current_step = self.window
        self.cash = self.init_balance
        self.shares = np.zeros(self.n_sym, dtype=np.float64)
        self.weights = np.zeros(self.n_sym, dtype=np.float64)
        self.weight_smoothed = np.zeros(self.n_sym, dtype=np.float64)
        self.last_trade_step = np.full(self.n_sym, -self.cooldown_steps, dtype=int)
        
        # Метрики
        self.portfolio_value_hist = [self.init_balance]
        self.peak_value = self.init_balance
        self.total_trades = 0
        self.returns_buffer = UnifiedBuffer(maxlen=100)
    
    def seed(self, seed: int):
        np.random.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_internal()
        return self._get_obs(), {}
    
    def _get_obs(self) -> np.ndarray:
        # История цен (window, n_sym)
        price_block = self.norm_prices[
            self.current_step - self.window:self.current_step
        ].flatten()
        
        # Текущие фичи (n_feat, n_sym)
        feat_block = []
        for f in self.feature_cols:
            arr = self.feature_data.get(f)
            if arr is not None:
                val = arr[self.current_step - 1]
                val = np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
                feat_block.extend(val.tolist())
            else:
                feat_block.extend([0.0] * self.n_sym)
        
        # Доля кэша (0-1)
        current_value = self._get_portfolio_value()
        cash_frac = np.array([self.cash / (current_value + 1e-9)])
        
        # Текущие веса в портфеле
        weights = self.weights
        
        # Объединяем все блоки
        obs = np.concatenate([price_block, np.array(feat_block), cash_frac, weights])
        
        # === ФИНАЛЬНАЯ ПРОВЕРКА: ===
        if not np.all(np.isfinite(obs)):
            print(f"❌ NaN/Inf detected in observation at step {self.current_step}")
            print(f"   Price block finite: {np.all(np.isfinite(price_block))}")
            print(f"   Feature block finite: {np.all(np.isfinite(feat_block))}")
            print(f"   Cash frac finite: {np.isfinite(cash_frac[0])}")
            print(f"   Weights finite: {np.all(np.isfinite(weights))}")
            raise ValueError("Observation contains NaN/Inf values!")
        
        return obs.astype(np.float32)
    
    def _get_portfolio_value(self) -> float:
        """Расчет текущей стоимости портфеля"""
        prices = self.prices[self.current_step]
        return self.cash + np.sum(self.shares * prices)
    
    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Нормализация действий с учетом лимитов"""
        # Клиппинг по границам
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Ограничение максимального веса на актив
        action = np.minimum(action, self.max_weight_per_asset)
        
        # Нормализация суммы весов (если включено)
        if self.normalize_action:
            total = np.sum(action)
            if total > 1.0 + 1e-9:
                action = action / total
        
        return action
    
    def step(self, action: np.ndarray) -> tuple:
        # Нормализация действия
        target_weights = self._normalize_action(action)
        
        # Применение cooldown
        cooldown_mask = (self.current_step - self.last_trade_step) < self.cooldown_steps
        target_weights[cooldown_mask] = self.weights[cooldown_mask]
        
        # Сглаживание весов (если включено)
        if self.weight_smoothing_alpha is not None:
            alpha = float(self.weight_smoothing_alpha)
            target_weights = alpha * target_weights + (1.0 - alpha) * self.weight_smoothed
            target_weights = self._normalize_action(target_weights)
        
        # Текущие цены и стоимость портфеля
        prices = self.prices[self.current_step]
        current_value = self._get_portfolio_value()
        
        # Расчет целевых позиций
        target_shares = (current_value * target_weights) / (prices + 1e-9)
        trade_shares = target_shares - self.shares
        
        # Ограничение размера сделки (max_trade_pct)
        max_trade_value = self.max_trade_pct * current_value
        max_trade_shares = max_trade_value / (prices + 1e-9)
        trade_shares = np.sign(trade_shares) * np.minimum(
            np.abs(trade_shares), max_trade_shares
        )
        
        # Фильтр минимального размера сделки
        trade_values = np.abs(trade_shares * prices)
        small_trade_mask = trade_values < self.min_trade_value
        trade_shares[small_trade_mask] = 0.0
        
        # Выполнение сделок с учетом комиссий и проскальзывания
        self._execute_trades(trade_shares, prices)
        
        # Обновление временного шага
        self.current_step += 1
        done = self.current_step >= len(self.dates) - 1
        
        # Обновление портфельных весов
        new_value = self._get_portfolio_value()
        self.weights = (self.shares * prices) / (new_value + 1e-9)
        self.weight_smoothed = self.weights.copy()
        
        # Обновление времени последней сделки
        traded_mask = np.abs(trade_shares) > 1e-9
        self.last_trade_step[traded_mask] = self.current_step
        
        # Расчет награды
        reward = self._calculate_reward(new_value, trade_values.sum())
        
        # Формирование информации для логирования
        info = {
            "portfolio_value": float(new_value),
            "cash": float(self.cash),
            "turnover": float(trade_values.sum() / (current_value + 1e-9)),
            "weights": self.weights.copy(),
            "drawdown": float((self.peak_value - new_value) / (self.peak_value + 1e-9)),
            "total_trades": int(self.total_trades),
        }
        
        # Финальные метрики при завершении эпизода
        if done:
            info["final_metrics"] = self._get_final_metrics()
        
        return self._get_obs(), float(reward), bool(done), False, info
    
    def _execute_trades(self, trade_shares: np.ndarray, prices: np.ndarray):
        """
        Выполнение сделок с учетом комиссий и проскальзывания
        Сначала продажи, затем покупки
        """
        # === ПРОДАЖИ (отрицательные trade_shares) ===
        sell_mask = trade_shares < 0
        if sell_mask.any():
            sell_qty = -trade_shares[sell_mask]
            sell_value = sell_qty * prices[sell_mask]
            
            # Комиссия и проскальзывание
            costs = sell_value * (self.comm + self.slip)
            net_proceeds = sell_value - costs
            
            # Обновляем позицию и кэш
            self.shares[sell_mask] -= sell_qty
            self.cash += net_proceeds.sum()
        
        # === ПОКУПКИ (положительные trade_shares) ===
        buy_mask = trade_shares > 0
        if buy_mask.any():
            buy_qty = trade_shares[buy_mask]
            buy_value = buy_qty * prices[buy_mask]
            costs = buy_value * (self.comm + self.slip)
            total_cost = buy_value + costs
            
            # Проверяем достаточность кэша
            if total_cost.sum() <= self.cash + 1e-9:
                # Полная покупка
                self.shares[buy_mask] += buy_qty
                self.cash -= total_cost.sum()
            else:
                # Покупаем пропорционально доступному кэшу
                scale = self.cash / (total_cost.sum() + 1e-9)
                scaled_qty = buy_qty * scale
                self.shares[buy_mask] += scaled_qty
                spent = (scaled_qty * prices[buy_mask] * (1 + self.comm + self.slip)).sum()
                self.cash -= spent
        
        # Запрещаем отрицательные позиции (ЕСЛИ ШОРТ НЕ РАЗРЕШЕН)
        if not self.allow_short:
            self.shares = np.maximum(self.shares, 0.0)
        
        # Считаем количество совершенных сделок
        self.total_trades += np.sum(np.abs(trade_shares) > 1e-9)
    
    def _calculate_reward(self, new_value: float, turnover: float) -> float:
        old_value = self.portfolio_value_hist[-1]
        
        # 1. Доходность
        pnl = (new_value - old_value) / old_value * 100.0
        
        # 2. Пенальти за просадку
        self.peak_value = max(self.peak_value, new_value)
        drawdown = (self.peak_value - new_value) / self.peak_value * 100.0
        
        # 3. Пенальти за оборот
        turnover_penalty = self.turnover_penalty * turnover * 100.0
        
        # 4. BONUS за SHARPE
        self.returns_buffer.append(pnl)
        returns = self.returns_buffer.get()
        sharpe_bonus = 0.0
        if len(returns) > 10:
            std = returns.std()
            if std > 1e-6:
                sharpe = (returns.mean() - self.risk_free_rate) / std
                sharpe_bonus = 0.1 * sharpe
        
        # 5. Награда
        reward = pnl - self.dd_penalty * drawdown - turnover_penalty + sharpe_bonus
        
        # 6. Клиппинг (±10% за шаг)
        reward = np.clip(reward, -10.0, 10.0)
        
        # 7. Обновляем историю портфеля
        self.portfolio_value_hist.append(new_value)
        
        return float(reward)
    
    def _get_final_metrics(self) -> Dict[str, float]:
        """Финальные метрики эпизода (Sharpe, Drawdown, Return)"""
        returns = np.diff(self.portfolio_value_hist) / np.array(self.portfolio_value_hist[:-1])
        
        # Sharpe ratio
        sharpe = 0.0
        if len(returns) > 1 and returns.std() > 1e-6:
            sharpe = np.sqrt(252) * (returns.mean() / returns.std())
        
        # Max drawdown
        peak = np.maximum.accumulate(self.portfolio_value_hist)
        max_dd = np.max((peak - self.portfolio_value_hist) / (peak + 1e-9))
        
        # Calmar ratio
        calmar = 0.0
        total_return = (self.portfolio_value_hist[-1] / self.init_balance - 1)
        if max_dd > 1e-6:
            calmar = total_return / max_dd
        
        return {
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "calmar_ratio": float(calmar),
            "total_return": float(total_return),
            "final_value": float(self.portfolio_value_hist[-1]),
            "total_trades": int(self.total_trades),
            "avg_trade_size": float(self.cash / max(1, self.total_trades)),
        }
    
    def render(self, mode="human"):
        """Рендеринг текущего состояния"""
        date = self.dates[self.current_step - 1]
        print(
            f"{date}: PV={self.portfolio_value_hist[-1]:,.2f} | "
            f"Cash={self.cash:,.2f} | "
            f"W={self.weights.round(3)}"
        )
