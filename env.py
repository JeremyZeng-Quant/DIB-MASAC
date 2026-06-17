# env.py
import os
import glob
import random
import numpy as np
import pandas as pd
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_file_path(pattern):
    search_dirs = [
        ".", 
        "..", 
        os.path.expanduser("~"), 
        os.path.join(os.path.expanduser("~"), "LOBench-A-share-processed"),
        os.path.join(os.path.expanduser("~"), "lob_bench_data")
    ]
    for d in search_dirs:
        matched = glob.glob(os.path.join(d, pattern))
        if matched:
            exclude_keywords = ["trajectory", "best", "model"]
            filtered = [m for m in matched if not any(k in os.path.basename(m).lower() for k in exclude_keywords)]
            if filtered:
                return os.path.abspath(filtered[0])
    return None

class HFMMEnvironment:
    def __init__(self, lob_data, kappa=0.0001, seq_len=5, max_inventory=50, action_scale=1.0, 
                 normalize_reward=False, dataset_type="", initial_capital=10000.0):
        self.lob_data = lob_data.copy().astype(np.float32)
        self.initial_capital = initial_capital
        self.dataset_type = dataset_type.lower()
        
        self.kappa = kappa
        self.seq_len = seq_len
        self.max_inventory = max_inventory
        self.action_scale = action_scale
        self.normalize_reward = normalize_reward
        
        if "binance" in self.dataset_type:
            self.maker_fee_rate = 0.00001
        else:
            self.maker_fee_rate = 0.00005

        sample_mid = (self.lob_data[0, 0] + self.lob_data[0, 2]) / 2.0
        self.prob_scale = max(0.01, sample_mid * 0.0001)
        self.reward_norm_base = max(1.0, self.initial_capital * 0.0005)
        self.reset()

    def reset(self):
        self.current_tick = self.seq_len
        self.cash = 0.0
        self.inventory = 0.0
        return self._get_state()

    def _get_state(self):
        state = self.lob_data[self.current_tick - self.seq_len: self.current_tick]
        state_normalized = state.copy().astype(np.float32)
        p1 = state[:, 0]
        p2 = state[:, 2]
        mid = (p1 + p2) / 2.0
        for j in range(10):
            state_normalized[:, j * 4] -= mid
            state_normalized[:, j * 4 + 2] -= mid
        return torch.tensor(state_normalized, dtype=torch.float32).unsqueeze(0).to(device)

    def step(self, action_mm, action_it):
        p1 = self.lob_data[self.current_tick, 0]
        p2 = self.lob_data[self.current_tick, 2]
        ask_1 = max(p1, p2)
        bid_1 = min(p1, p2)
        mid_price = (ask_1 + bid_1) / 2.0

        tick_size = 0.01
        raw_delta_plus = abs(action_mm[0, 0].item()) * self.action_scale
        raw_delta_minus = abs(action_mm[0, 1].item()) * self.action_scale

        delta_plus = max(round(raw_delta_plus / tick_size) * tick_size, tick_size)
        delta_minus = max(round(raw_delta_minus / tick_size) * tick_size, tick_size)

        nu = action_it[0, 0].item() if isinstance(action_it, torch.Tensor) else action_it

        prob_fill_ask = np.exp(-1.5 * (delta_plus / self.prob_scale) - 0.5 * nu)
        prob_fill_bid = np.exp(-1.5 * (delta_minus / self.prob_scale) + 0.5 * nu)

        fill_ask = (self.inventory > -self.max_inventory) and (np.random.rand() < prob_fill_ask)
        fill_bid = (self.inventory < self.max_inventory) and (np.random.rand() < prob_fill_bid)

        prev_portfolio_value = self.cash + self.inventory * mid_price

        if fill_ask:
            self.inventory -= 1.0
            exec_price = mid_price + delta_plus
            fee = exec_price * self.maker_fee_rate
            self.cash += (exec_price - fee)
        if fill_bid:
            self.inventory += 1.0
            exec_price = mid_price - delta_minus
            fee = exec_price * self.maker_fee_rate
            self.cash -= (exec_price + fee)

        normalized_inv = abs(self.inventory) / self.max_inventory
        r_penalty = - 1.0 * (normalized_inv ** 4) * np.exp(2.0 * normalized_inv)
        r_penalty = np.clip(r_penalty, -2.0, 0.0)

        liquidation_fee = 0.0
        if abs(self.inventory) >= self.max_inventory * 0.95:
            liquidation_fee = -1.0

        self.current_tick += 1
        next_p1 = self.lob_data[self.current_tick, 0]
        next_p2 = self.lob_data[self.current_tick, 2]
        next_mid_price = (max(next_p1, next_p2) + min(next_p1, next_p2)) / 2.0
        
        current_portfolio_value = self.cash + self.inventory * next_mid_price
        is_bankrupt = (current_portfolio_value <= -0.9 * self.initial_capital)
        raw_r_wealth = current_portfolio_value - prev_portfolio_value

        if self.normalize_reward:
            r_wealth = (raw_r_wealth / max(mid_price, 1e-6)) * 1000.0
        else:
            r_wealth = raw_r_wealth

        r_wealth = np.clip(r_wealth, -100.0, 100.0)
        if is_bankrupt:
            liquidation_fee -= 5.0

        total_raw_reward = r_wealth + r_penalty + liquidation_fee
        reward_scaled = np.clip(total_raw_reward / self.reward_norm_base, -1.0, 1.0)
        
        reward = torch.tensor([[reward_scaled]], dtype=torch.float32).to(device)
        next_state = self._get_state()
        done = (self.current_tick >= len(self.lob_data) - 2) or is_bankrupt

        return next_state, reward, done

def load_lob_data(filepath, dataset_type="fi2010", max_ticks=None, is_real_target=True):
    if filepath is not None and os.path.exists(filepath):
        print(f"   📊 加载高频 LOB 数据: {filepath} ...")
        if dataset_type == "a_share":
            cols = []
            for i in range(1, 11):
                cols.extend([f'AskPrice{i}', f'AskVolume{i}', f'BidPrice{i}', f'BidVolume{i}'])
            df = pd.read_csv(filepath, usecols=cols, nrows=max_ticks)
            raw_data_t = df[cols].values
        elif filepath.endswith('.pt'):
            raw_data_t = torch.load(filepath, map_location='cpu', weights_only=False)
            if isinstance(raw_data_t, dict):
                for k in ['train', 'val', 'dataset', 'data', 'x']:
                    if k in raw_data_t:
                        raw_data_t = raw_data_t[k]
                        break
            if torch.is_tensor(raw_data_t):
                raw_data_t = raw_data_t.numpy()
            if len(raw_data_t.shape) == 3:
                raw_data_t = raw_data_t[:, -1, :]
        else:
            raw_data_t = np.loadtxt(filepath)
            raw_data_t = raw_data_t.T if raw_data_t.shape[0] == 149 else raw_data_t

        raw_data_t = np.asarray(raw_data_t, dtype=np.float32)
        if max_ticks is not None and len(raw_data_t) > max_ticks:
            raw_data_t = raw_data_t[:max_ticks]
        return raw_data_t[:, :40]
    else:
        raise FileNotFoundError(f"未在当前目录下找到所需的 {filepath} 文件，请检查该文件是否存在！")