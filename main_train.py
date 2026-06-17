import os
import gc
import glob
import csv
import random
import argparse
import traceback
import warnings
import time
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

# 🎯 升级：导入绘图库，用于自动生成和保存实验图表
import matplotlib
matplotlib.use('Agg')  # 在后台静默绘图，无需GUI界面，防止服务器报错
import matplotlib.pyplot as plt

# 忽略不必要的冗余警告
warnings.filterwarnings("ignore")

# ========== 全局超参数（直接作为论文参数表复制） ==========
TAU = 0.995  # 目标网络软更新系数
LR = 3e-5  # 统一优化器学习率
GAMMA = 0.99  # 折扣因子
CLIP_EPS = 0.2  # PPO 裁剪范围
ALPHA_0 = 0.2  # 默认 SAC 熵正则化系数 (在主循环中会针对高费率市场自适应收紧)
KL_COEF = 1e-4  # DIB 表征瓶颈约束系数
BETA = 0.1  # DIB 辅助预测权重
G_MAX = 1.0  # 梯度裁剪硬限
MAX_INVENTORY = 50  # 绝对持仓限制
SEQ_LEN = 5  # 做市决策回溯窗口

BATCH_SIZE = 1024  
UPDATE_EVERY = 250  # ✅ 从500改250，减少CPU等待GPU的空闲时间

# === 均值-方差效用持久化配置文件路径 ===
PROGRESS_LOG_FILE = "aaai_experiment_progress.csv"

# 🎯 升级：增加耗时与分区路径等新字段，避免参数传递崩溃并提高可读性
def save_result_to_csv(file_path, dataset, mode, seed, utility, sharpe, inv_var, max_dd, trades, start_time_str, elapsed_sec, is_our, folder_name):
    """自动将跑完的每组种子实验追加保存到硬盘，增加时间追踪与分区辨识"""
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "dataset", "mode", "seed", "utility", "sharpe", 
                "inv_variance", "max_dd", "trades", "start_time", 
                "elapsed_sec", "is_our_method", "saved_folder"
            ])
        writer.writerow([
            dataset, mode, seed, 
            f"{utility:.4f}", f"{sharpe:.4f}", f"{inv_var:.4f}", f"{max_dd:.4f}", trades,
            start_time_str, f"{elapsed_sec:.2f}", "YES" if is_our else "NO", folder_name
        ])

# 🎯 升级：新增自动绘图并保存函数（保存到对应分区文件夹下，且带唯一时间戳）
def plot_and_save_trajectory(portfolio_history, val_inventories, mode, dataset_type, seed, output_dir, timestamp):
    """自动绘制资产净值曲线与持仓变化图，并保存至特定分区的硬盘目录"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # 绘制资产净值变化
    ax1.plot(portfolio_history, label='Portfolio Value (NAV)', color='tab:blue', lw=1.5)
    ax1.set_ylabel('NAV / Capital')
    ax1.set_title(f'Trajectory Analysis - {mode.upper()} on {dataset_type.upper()} (Seed {seed})')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper left')
    
    # 绘制持仓仓位变化
    ax2.plot(val_inventories, label='Inventory (Position)', color='tab:orange', lw=1.2)
    ax2.axhline(0, color='black', linestyle=':', alpha=0.5)
    ax2.set_ylabel('Position / Units')
    ax2.set_xlabel('Trading Ticks')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"trajectory_plot_{mode}_{dataset_type}_seed{seed}_{timestamp}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"   📊 [图表卫士] 成功将回测轨迹图自动保存至: {plot_path}", flush=True)

# ====================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.distributions.Distribution.set_default_validate_args(False)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

EVALUATE_MODE = True
MAX_TICKS = None if EVALUATE_MODE else 50000
MAX_EPOCHS = 30 if EVALUATE_MODE else 5

TICK_SIZES = {
    "fi2010": 0.0001,
    "lob_bench_goog_real": 0.01,
    "lob_bench_goog_synth": 0.01,  
    "lob_bench_intc_real": 0.01,
    "lob_bench_intc_synth": 0.01,  
    "binance_low_vol": 0.1,  
    "binance_high_vol": 0.1,
    "trades_lob_tsla": 0.01,
    "trades_lob_intc": 0.01,
    "a_share_sz000001": 0.01,
    "a_share_sz000651": 0.01,
    "a_share_sz002415": 0.01,
    "a_share_sz300147": 0.01,
}

device = torch.device("cpu")  


# ====================== GPU 驻留 Replay Buffer ======================
class ReplayBuffer:
    def __init__(self, capacity=50000, device=torch.device("cpu")):
        self.capacity = capacity
        self.device = device
        self.buffer = []
        self.position = 0

    def push(self, state, action_mm, action_it, reward, next_state, done):
        state_dev = state.detach().to(self.device)
        action_mm_dev = action_mm.detach().to(self.device)
        action_it_dev = action_it.detach().to(self.device) if torch.is_tensor(action_it) else action_it
        reward_dev = reward.detach().to(self.device)
        next_state_dev = next_state.detach().to(self.device)

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (state_dev, action_mm_dev, action_it_dev, reward_dev, next_state_dev, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, action_mms, action_its, rewards, next_states, dones = zip(*batch)

        states = torch.cat(states, dim=0)
        action_mms = torch.cat(action_mms, dim=0)
        rewards = torch.cat(rewards, dim=0)
        next_states = torch.cat(next_states, dim=0)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        if all(torch.is_tensor(x) for x in action_its):
            action_its_cat = torch.cat(action_its, dim=0)
        else:
            action_its_cat = torch.zeros((batch_size, 1), device=self.device)

        return states, action_mms, action_its_cat, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


def find_file_path(pattern):
    search_dirs = [
        ".",
        "..",
        os.path.expanduser("~"),
        os.path.join(os.path.expanduser("~"), "LOBench-A-share-processed")
    ]
    for d in search_dirs:
        matched = glob.glob(os.path.join(d, pattern))
        if matched:
            exclude_keywords = ["trajectory", "best", "model"]
            if EVALUATE_MODE:
                exclude_keywords.append("toy")
            filtered = [m for m in matched if not any(k in os.path.basename(m).lower() for k in exclude_keywords)]
            if filtered:
                return os.path.abspath(filtered[0])
    return None


# ======================================================================
# 1. Spatio-Temporal DIB Encoder
# ======================================================================

class AdvancedDIBLOBEncoder(nn.Module):
    def __init__(self, channels=1, latent_dim=32):
        super(AdvancedDIBLOBEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.attn = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(32 * 41, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_log_var = nn.Linear(64, latent_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, L_t, deterministic=False):
        h_conv = self.conv(L_t)  
        h_conv = h_conv.transpose(1, 2)  
        attn_out, _ = self.attn(h_conv, h_conv, h_conv)
        h_flat = attn_out.reshape(attn_out.size(0), -1)  
        
        h = self.fc(h_flat)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)

        mu = torch.clamp(mu, -10.0, 10.0)
        log_var = torch.clamp(log_var, -8.0, 4.0)

        if deterministic:
            s_t = mu
        else:
            s_t = self.reparameterize(mu, log_var)
        return s_t, mu, log_var



class BiGRUDIBEncoder(nn.Module):
    """BiGRU时序瓶颈编码器 + DIB信息瓶颈，替代Conv+Attention结构"""
    def __init__(self, latent_dim=32):
        super(BiGRUDIBEncoder, self).__init__()
        # 特征提取：5通道 -> 32通道，保持与AdvancedDIBLOBEncoder兼容
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        # 双向GRU：输入32维，hidden 32，双向输出64
        self.bigru = nn.GRU(
            input_size=32,
            hidden_size=16,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        self.norm = nn.LayerNorm(32)
        self.fc = nn.Sequential(
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_log_var = nn.Linear(64, latent_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, L_t, deterministic=False):
        # L_t: (B, 5, 41)
        h_conv = self.conv(L_t)           # (B, 32, 41)
        h_seq = h_conv.transpose(1, 2)   # (B, 41, 32)
        gru_out, _ = self.bigru(h_seq)   # (B, 41, 64)
        h_pool = gru_out.mean(dim=1)     # (B, 64) — 时序mean pooling
        h_norm = self.norm(h_pool)
        h = self.fc(h_norm)
        mu = torch.clamp(self.fc_mu(h), -10.0, 10.0)
        log_var = torch.clamp(self.fc_log_var(h), -8.0, 4.0)
        if deterministic:
            return mu, mu, log_var
        return self.reparameterize(mu, log_var), mu, log_var


class TimesFMEncoder(nn.Module):
    def __init__(self, patch_len=1, d_model=128, latent_dim=32):
        super(TimesFMEncoder, self).__init__()
        self.patch_proj = nn.Linear(41 * patch_len, d_model)
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, d_model)
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, latent_dim)

    def forward(self, L_t):
        batch_size, seq_len, features = L_t.shape
        x_proj = self.patch_proj(L_t)
        attn_out, _ = self.self_attn(x_proj, x_proj, x_proj)
        x_norm = self.layer_norm(x_proj + attn_out)
        ffn_out = self.ffn(x_norm)
        out = self.layer_norm(x_norm + ffn_out)
        s_t = self.fc_out(out.mean(dim=1))
        return s_t


class MomentEncoder(nn.Module):
    def __init__(self, d_model=128, latent_dim=32):
        super(MomentEncoder, self).__init__()
        self.projection = nn.Linear(41, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=256, batch_first=True,
                                       norm_first=True),
            num_layers=2
        )
        self.fc_out = nn.Linear(d_model, latent_dim)

    def forward(self, L_t):
        batch_size, seq_len, features = L_t.shape
        x_proj = self.projection(L_t)
        out = self.transformer(x_proj)
        s_t = self.fc_out(out.mean(dim=1))
        return s_t


class TimerXLEncoder(nn.Module):
    def __init__(self, d_model=128, latent_dim=32):
        super(TimerXLEncoder, self).__init__()
        self.patch_proj = nn.Linear(41, d_model)
        self.causal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=256, batch_first=True),
            num_layers=3
        )
        self.fc_out = nn.Linear(d_model, latent_dim)

    def forward(self, L_t):
        batch_size, seq_len, features = L_t.shape
        x_proj = self.patch_proj(L_t)
        out = self.causal_transformer(x_proj)
        s_t = self.fc_out(out[:, -1, :])
        return s_t


class DeepLOBEncoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(DeepLOBEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.lstm = nn.LSTM(input_size=64 * 41, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, latent_dim)

    def forward(self, L_t):
        batch_size, seq_len, features = L_t.shape
        x = L_t.view(batch_size * seq_len, 1, features)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(batch_size, seq_len, -1)
        out, _ = self.lstm(x)
        s_t = self.fc(out[:, -1, :])
        return s_t


# ======================================================================
# 2. Decentralized Actors & Centralized Critic
# ======================================================================
class GaussianActor(nn.Module):
    def __init__(self, state_dim, action_dim, action_limit=1.0):
        super(GaussianActor, self).__init__()
        self.action_limit = action_limit
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)

        nn.init.uniform_(self.mu_layer.weight, -1e-4, 1e-4)
        nn.init.constant_(self.mu_layer.bias, 0.0)

    def forward(self, state, deterministic=False, with_logprob=True):
        features = self.net(state)
        mu = self.mu_layer(features)
        log_std = self.log_std_layer(features)

        mu = torch.clamp(mu, -5.0, 5.0)
        log_std = torch.clamp(log_std, -8.0, 1.0)
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        if deterministic:
            action = mu
        else:
            action = dist.rsample()

        action_tanh = torch.tanh(action)
        scaled_action = self.action_limit * action_tanh

        if with_logprob:
            log_prob = dist.log_prob(action) - torch.log(self.action_limit * (1 - action_tanh.pow(2)) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            log_prob = torch.clamp(log_prob, -50.0, 10.0)
        else:
            log_prob = None

        return scaled_action, log_prob


class CentralizedCritic(nn.Module):
    def __init__(self, state_dim, action_dim_mm, action_dim_it):
        super(CentralizedCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim_mm + action_dim_it, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action_mm, action_it):
        x = torch.cat([state, action_mm, action_it], dim=-1)
        return self.net(x)


# ======================================================================
# 3. 策略与对比基线算法类与融合映射
# ======================================================================

def blend_action(action_mm_raw, action_as, action_scale, max_gate=1.0, min_val=0.0, delta_scale=3.0):
    delta_rl = action_mm_raw[..., :2] * (delta_scale * action_scale)
    
    g_gate = ((action_mm_raw[..., 2:3] + 1.0) / 2.0) if action_mm_raw.shape[-1] > 2 else torch.ones(action_mm_raw.shape[0], 1, device=action_mm_raw.device)
    g_gate = torch.clamp(g_gate, max=max_gate)
    
    blended = action_as + g_gate * delta_rl
    return torch.clamp(blended, min=min_val, max=10.0)


class AvellanedaStoikovAgent:
    def __init__(self, gamma=0.1, kappa=1.5, window_size=100):
        self.gamma = gamma
        self.kappa = kappa
        self.window_size = window_size
        self.mid_history = []

    def get_action(self, mid_price, inventory, tick_size, max_offset=3.0):
        self.mid_history.append(mid_price)
        if len(self.mid_history) > self.window_size:
            self.mid_history.pop(0)

        if len(self.mid_history) < 2:
            sigma_log = 0.0001
        else:
            log_returns = np.diff(np.log(self.mid_history))
            sigma_log = np.std(log_returns) if np.std(log_returns) > 1e-6 else 0.0001

        sigma = mid_price * sigma_log
        adaptive_kappa = self.kappa / tick_size

        r = mid_price - inventory * self.gamma * (sigma ** 2)
        spread = self.gamma * (sigma ** 2) + (2 / self.gamma) * np.log(1 + self.gamma / adaptive_kappa)

        my_ask = r + spread / 2.0
        my_bid = r - spread / 2.0

        act_mm_0 = np.clip((my_ask - mid_price) / tick_size, 0.0, max_offset)
        act_mm_1 = np.clip((mid_price - my_bid) / tick_size, 0.0, max_offset)
        return torch.tensor([[act_mm_0, act_mm_1]], dtype=torch.float32).to(device)


class HoStollAgent:
    def __init__(self, gamma=0.1, window_size=100):
        self.gamma = gamma
        self.window_size = window_size
        self.mid_history = []

    def get_action(self, mid_price, inventory, tick_size, max_offset=3.0):
        self.mid_history.append(mid_price)
        if len(self.mid_history) > self.window_size:
            self.mid_history.pop(0)

        if len(self.mid_history) < 2:
            sigma_log = 0.0001
        else:
            log_returns = np.diff(np.log(self.mid_history))
            sigma_log = np.std(log_returns) if np.std(log_returns) > 1e-6 else 0.0001

        sigma = mid_price * sigma_log

        r_ask = mid_price + (1 + 2 * inventory) * 0.5 * self.gamma * (sigma ** 2)
        r_bid = mid_price - (1 - 2 * inventory) * 0.5 * self.gamma * (sigma ** 2)

        act_mm_0 = np.clip((r_ask - mid_price) / tick_size, 0.0, max_offset)
        act_mm_1 = np.clip((mid_price - r_bid) / tick_size, 0.0, max_offset)
        return torch.tensor([[act_mm_0, act_mm_1]], dtype=torch.float32).to(device)


class SinglePPOAgent:
    def __init__(self, state_dim=32, action_dim_mm=2, lr=LR, clip_eps=CLIP_EPS, gamma=GAMMA, G_max=G_MAX):
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.G_max = G_max

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5 * 41, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim)
        ).to(device)

        self.actor_mm = GaussianActor(state_dim, action_dim_mm).to(device)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(device)

        self.opt = optim.Adam(
            list(self.encoder.parameters()) +
            list(self.actor_mm.parameters()) +
            list(self.critic.parameters()),
            lr=lr
        )

    def train_step(self, L_t, action_mm_hist, action_it_hist, r_t, L_next, done):
        s_t = self.encoder(L_t)
        v_t = self.critic(s_t)

        with torch.no_grad():
            s_next = self.encoder(L_next)
            v_next = self.critic(s_next)
            td_target = r_t + (1 - done) * self.gamma * v_next
            advantage = td_target - v_t

        _, log_prob_old_mm = self.actor_mm(s_t.detach(), deterministic=False, with_logprob=True)
        _, log_prob_new_mm = self.actor_mm(s_t, deterministic=False, with_logprob=True)

        ratio_mm = torch.exp(log_prob_new_mm - log_prob_old_mm.detach())
        surr1_mm = ratio_mm * advantage.detach()
        surr2_mm = torch.clamp(ratio_mm, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantage.detach()

        actor_mm_loss = -torch.min(surr1_mm, surr2_mm).mean()
        critic_loss = F.mse_loss(v_t, td_target.detach())
        total_loss = actor_mm_loss + 0.5 * critic_loss

        self.opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) +
            list(self.actor_mm.parameters()) +
            list(self.critic.parameters()),
            self.G_max
        )
        self.opt.step()

        return {
            "critic_loss": critic_loss.item(),
            "actor_mm_loss": actor_mm_loss.item(),
            "actor_it_loss": 0.0,
            "dib_loss": 0.0
        }


class HierarchicalRLAgent:
    def __init__(self, state_dim=32, action_dim_mm=2, lr=LR, gamma=GAMMA, G_max=G_MAX):
        self.gamma = gamma
        self.G_max = G_max
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5 * 41, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim)
        ).to(device)

        self.manager = GaussianActor(state_dim, action_dim=1).to(device)
        self.actor_mm = GaussianActor(state_dim + 1, action_dim_mm).to(device)

        self.critic_1 = CentralizedCritic(state_dim, action_dim_mm, 1).to(device)
        self.critic_2 = CentralizedCritic(state_dim, action_dim_mm, 1).to(device)

        self.opt = optim.Adam(
            list(self.encoder.parameters()) +
            list(self.manager.parameters()) +
            list(self.actor_mm.parameters()) +
            list(self.critic_1.parameters()) +
            list(self.critic_2.parameters()),
            lr=lr
        )

    def train_step(self, L_t, action_mm_hist, action_it_hist, r_t, L_next, done):
        s_t = self.encoder(L_t)
        s_next = self.encoder(L_next)

        goal, log_prob_goal = self.manager(s_t)
        s_worker = torch.cat([s_t, goal], dim=-1)

        with torch.no_grad():
            next_goal, _ = self.manager(s_next)
            s_worker_next = torch.cat([s_next, next_goal], dim=-1)
            next_act_mm, _ = self.actor_mm(s_worker_next)
            target_Q1 = self.critic_1(s_next, next_act_mm, torch.zeros_like(action_it_hist))
            target_Q2 = self.critic_2(s_next, next_act_mm, torch.zeros_like(action_it_hist))
            target_Q = torch.min(target_Q1, target_Q2)
            y = r_t + (1 - done) * self.gamma * target_Q

        q1 = self.critic_1(s_t, action_mm_hist, action_it_hist)
        q2 = self.critic_2(s_t, action_mm_hist, action_it_hist)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        act_mm, log_prob_mm = self.actor_mm(s_worker)
        q1_pi_worker = self.critic_1(s_t, act_mm, torch.zeros_like(action_it_hist))
        q2_pi_worker = self.critic_2(s_t, act_mm, torch.zeros_like(action_it_hist))
        q_pi_worker = torch.min(q1_pi_worker, q2_pi_worker)
        actor_mm_loss = (ALPHA_0 * log_prob_mm - q_pi_worker).mean()

        q1_pi_manager = self.critic_1(s_t, act_mm.detach(), goal)
        q2_pi_manager = self.critic_2(s_t, act_mm.detach(), goal)
        q_pi_manager = torch.min(q1_pi_manager, q2_pi_manager)
        manager_loss = (ALPHA_0 * log_prob_goal - q_pi_manager).mean()

        total_loss = critic_loss + actor_mm_loss + manager_loss

        self.opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) +
            list(self.manager.parameters()) +
            list(self.actor_mm.parameters()) +
            list(self.critic_1.parameters()) +
            list(self.critic_2.parameters()),
            self.G_max
        )
        self.opt.step()

        return {
            "critic_loss": critic_loss.item(),
            "actor_mm_loss": actor_mm_loss.item(),
            "actor_it_loss": manager_loss.item(),
            "dib_loss": 0.0
        }


class StandardMAPPO:
    def __init__(self, state_dim=32, action_dim_mm=2, action_dim_it=1, lr=LR, clip_eps=CLIP_EPS, gamma=GAMMA,
                 G_max=G_MAX):
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.G_max = G_max

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5 * 41, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim)
        ).to(device)

        self.actor_mm = GaussianActor(state_dim, action_dim_mm).to(device)
        self.actor_it = GaussianActor(state_dim, action_dim_it).to(device)

        self.critic = CentralizedCritic(state_dim, action_dim_mm, action_dim_it).to(device)

        self.opt = optim.Adam(
            list(self.encoder.parameters()) +
            list(self.actor_mm.parameters()) +
            list(self.actor_it.parameters()) +
            list(self.critic.parameters()),
            lr=lr
        )

    def train_step(self, L_t, action_mm_hist, action_it_hist, r_t, L_next, done):
        s_t = self.encoder(L_t)
        s_next = self.encoder(L_next)

        v_t = self.critic(s_t, action_mm_hist, action_it_hist)

        with torch.no_grad():
            next_act_mm, _ = self.actor_mm(s_next)
            next_act_it, _ = self.actor_it(s_next)
            v_next = self.critic(s_next, next_act_mm, next_act_it)
            td_target = r_t + (1 - done) * self.gamma * v_next
            advantage = td_target - v_t

        _, log_prob_old_mm = self.actor_mm(s_t.detach(), deterministic=False, with_logprob=True)
        _, log_prob_new_mm = self.actor_mm(s_t, deterministic=False, with_logprob=True)

        ratio_mm = torch.exp(log_prob_new_mm - log_prob_old_mm.detach())
        surr1_mm = ratio_mm * advantage.detach()
        surr2_mm = torch.clamp(ratio_mm, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantage.detach()
        actor_mm_loss = -torch.min(surr1_mm, surr2_mm).mean()

        _, log_prob_old_it = self.actor_it(s_t.detach(), deterministic=False, with_logprob=True)
        _, log_prob_new_it = self.actor_it(s_t, deterministic=False, with_logprob=True)
        ratio_it = torch.exp(log_prob_new_it - log_prob_old_it.detach())
        surr1_it = ratio_it * advantage.detach()
        surr2_it = torch.clamp(ratio_it, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantage.detach()
        actor_it_loss = -torch.min(surr1_it, surr2_it).mean()

        critic_loss = F.mse_loss(v_t, td_target.detach())
        total_loss = actor_mm_loss + actor_it_loss + 0.5 * critic_loss

        self.opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) +
            list(self.actor_mm.parameters()) +
            list(self.actor_it.parameters()) +
            list(self.critic.parameters()),
            self.G_max
        )
        self.opt.step()

        return {
            "critic_loss": critic_loss.item(),
            "actor_mm_loss": actor_mm_loss.item(),
            "actor_it_loss": actor_it_loss.item(),
            "dib_loss": 0.0
        }


class StabilizedAdversarialMASAC(nn.Module):
    def __init__(self, state_dim=32, action_dim_mm=3, action_dim_it=1, lr=LR, beta=BETA, alpha_0=ALPHA_0, gamma=GAMMA,
                 G_max=G_MAX, kl_coef=KL_COEF, ablation_mode=None):
        super(StabilizedAdversarialMASAC, self).__init__()
        self.gamma = gamma
        self.beta = beta
        self.alpha_0 = alpha_0
        self.G_max = G_max
        self.ablation_mode = ablation_mode  # None / "no_dib" / "no_it"
        self.kl_coef = 0.0 if ablation_mode == "no_dib" else kl_coef

        self.encoder = AdvancedDIBLOBEncoder(channels=1, latent_dim=state_dim).to(device)
        
        self.actor_it = GaussianActor(state_dim, action_dim_it).to(device)
        self.actor_mm = GaussianActor(state_dim + action_dim_it, 2).to(device)
        
        self.critic_1 = CentralizedCritic(state_dim, action_dim_mm, action_dim_it).to(device)
        self.critic_2 = CentralizedCritic(state_dim, action_dim_mm, action_dim_it).to(device)

        self.reward_decoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(device)

        self.encoder_target = AdvancedDIBLOBEncoder(channels=1, latent_dim=state_dim).to(device)
        self.encoder_target.load_state_dict(self.encoder.state_dict())

        self.critic_1_target = CentralizedCritic(state_dim, action_dim_mm, action_dim_it).to(device)
        self.critic_2_target = CentralizedCritic(state_dim, action_dim_mm, action_dim_it).to(device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.critic_opt = optim.Adam(
            list(self.critic_1.parameters()) +
            list(self.critic_2.parameters()) +
            list(self.encoder.parameters()) +
            list(self.reward_decoder.parameters()),
            lr=lr
        )
        self.actor_mm_opt = optim.Adam(self.actor_mm.parameters(), lr=lr)
        self.actor_it_opt = optim.Adam(self.actor_it.parameters(), lr=lr * 0.33)

    def train_step(self, L_t, action_mm_hist, action_it_hist, r_t, L_next, done):
        s_t, mu, log_var = self.encoder(L_t)

        with torch.no_grad():
            s_next, _, _ = self.encoder_target(L_next)

            next_act_it, log_prob_next_it = self.actor_it(s_next)
            s_worker_next = torch.cat([s_next, next_act_it], dim=-1)
            next_act_mm, log_prob_next_mm = self.actor_mm(s_worker_next)

            log_prob_next_mm = torch.clamp(log_prob_next_mm, -20.0, 10.0)
            log_prob_next_it = torch.clamp(log_prob_next_it, -20.0, 10.0)

            target_Q1 = self.critic_1_target(s_next, next_act_mm, next_act_it)
            target_Q2 = self.critic_2_target(s_next, next_act_mm, next_act_it)
            target_Q = torch.min(target_Q1, target_Q2)

            current_Q_conservative = torch.min(
                self.critic_1(s_t.detach(), action_mm_hist, action_it_hist),
                self.critic_2(s_t.detach(), action_mm_hist, action_it_hist)
            )
            exploitability = torch.abs(target_Q - current_Q_conservative).detach()
            alpha = self.alpha_0 * torch.exp(-0.05 * exploitability)
            alpha = torch.clamp(alpha, 0.005, self.alpha_0)

            y = r_t + (1 - done) * self.gamma * (target_Q - alpha * (log_prob_next_mm + log_prob_next_it))
            y = torch.clamp(y, -50.0, 50.0)

        q1 = self.critic_1(s_t, action_mm_hist, action_it_hist)
        q2 = self.critic_2(s_t, action_mm_hist, action_it_hist)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1).mean()
        kl_div = torch.clamp(kl_div, 1e-6, 20.0)
        pred_reward = self.reward_decoder(s_t)
        prediction_loss = F.mse_loss(pred_reward, r_t)

        if self.ablation_mode == "no_dib":
            dib_loss = self.beta * prediction_loss  # 消融DIB：去除KL约束，仅保留reward预测
        else:
            dib_loss = self.kl_coef * kl_div + self.beta * prediction_loss
        total_critic_loss = critic_loss + dib_loss

        self.critic_opt.zero_grad()
        total_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.critic_1.parameters()) +
            list(self.critic_2.parameters()) +
            list(self.encoder.parameters()) +
            list(self.reward_decoder.parameters()),
            self.G_max
        )
        self.critic_opt.step()

        s_t_detached = s_t.detach()

        act_it, log_prob_it = self.actor_it(s_t_detached)
        
        s_worker = torch.cat([s_t_detached, act_it.detach()], dim=-1)  
        act_mm, log_prob_mm = self.actor_mm(s_worker)

        q1_pi_mm = self.critic_1(s_t_detached, act_mm, act_it.detach())
        q2_pi_mm = self.critic_2(s_t_detached, act_mm, act_it.detach())
        q_pi_mm = torch.min(q1_pi_mm, q2_pi_mm)

        actor_mm_loss = (self.alpha_0 * log_prob_mm - q_pi_mm).mean()
        self.actor_mm_opt.zero_grad()
        actor_mm_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_mm.parameters(), self.G_max)
        self.actor_mm_opt.step()

        act_mm_for_it, _ = self.actor_mm(torch.cat([s_t_detached, act_it], dim=-1))
        q1_pi_it = self.critic_1(s_t_detached, act_mm_for_it.detach(), act_it)
        q2_pi_it = self.critic_2(s_t_detached, act_mm_for_it.detach(), act_it)
        q_pi_it = torch.min(q1_pi_it, q2_pi_it)

        actor_it_loss = (self.alpha_0 * log_prob_it - q_pi_it).mean()
        self.actor_it_opt.zero_grad()
        actor_it_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_it.parameters(), self.G_max)
        self.actor_it_opt.step()

        for param, target_param in zip(self.encoder.parameters(), self.encoder_target.parameters()):
            target_param.data.copy_(TAU * target_param.data + (1 - TAU) * param.data)
        for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
            target_param.data.copy_(TAU * target_param.data + (1 - TAU) * param.data)
        for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
            target_param.data.copy_(TAU * target_param.data + (1 - TAU) * param.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_mm_loss": actor_mm_loss.item(),
            "actor_it_loss": actor_it_loss.item(),
            "dib_loss": dib_loss.item()
        }


class GeneralizedSACBaseline(nn.Module):
    def __init__(self, encoder_type="sac", state_dim=32, action_dim_mm=2, lr=LR, alpha_0=ALPHA_0, gamma=GAMMA,
                 G_max=G_MAX):
        super(GeneralizedSACBaseline, self).__init__()
        self.gamma = gamma
        self.alpha_0 = alpha_0
        self.G_max = G_max

        if encoder_type == "timesfm":
            self.encoder = TimesFMEncoder(latent_dim=state_dim).to(device)
        elif encoder_type == "deeplob":
            self.encoder = DeepLOBEncoder(latent_dim=state_dim).to(device)
        elif encoder_type == "moment":
            self.encoder = MomentEncoder(latent_dim=state_dim).to(device)
        elif encoder_type == "timer_xl":
            self.encoder = TimerXLEncoder(latent_dim=state_dim).to(device)
        else:
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(5 * 41, 128),
                nn.ReLU(),
                nn.Linear(128, state_dim)
            ).to(device)

        self.actor_mm = GaussianActor(state_dim, action_dim_mm).to(device)

        self.critic_1 = CentralizedCritic(state_dim, action_dim_mm, 1).to(device)
        self.critic_2 = CentralizedCritic(state_dim, action_dim_mm, 1).to(device)

        self.critic_1_target = CentralizedCritic(state_dim, action_dim_mm, 1).to(device)
        self.critic_2_target = CentralizedCritic(state_dim, action_dim_mm, 1).to(device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.critic_opt = optim.Adam(
            list(self.critic_1.parameters()) +
            list(self.critic_2.parameters()) +
            list(self.encoder.parameters()),
            lr=lr
        )
        self.actor_mm_opt = optim.Adam(self.actor_mm.parameters(), lr=lr)

    def train_step(self, L_t, action_mm_hist, action_it_hist, r_t, L_next, done):
        s_t = self.encoder(L_t)

        with torch.no_grad():
            s_next = self.encoder(L_next)
            next_act_mm, log_prob_next_mm = self.actor_mm(s_next)
            next_act_it = torch.zeros_like(action_it_hist)

            target_Q1 = self.critic_1_target(s_next, next_act_mm, next_act_it)
            target_Q2 = self.critic_2_target(s_next, next_act_mm, next_act_it)
            target_Q = torch.min(target_Q1, target_Q2)

            y = r_t + (1 - done) * self.gamma * (target_Q - self.alpha_0 * log_prob_next_mm)
            y = torch.clamp(y, -50.0, 50.0)

        q1 = self.critic_1(s_t, action_mm_hist, action_it_hist)
        q2 = self.critic_2(s_t, action_mm_hist, action_it_hist)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.critic_1.parameters()) +
            list(self.critic_2.parameters()) +
            list(self.encoder.parameters()),
            self.G_max
        )
        self.critic_opt.step()

        s_t_detached = s_t.detach()
        act_mm, log_prob_mm = self.actor_mm(s_t_detached)

        q1_pi_mm = self.critic_1(s_t_detached, act_mm, torch.zeros_like(action_it_hist))
        q2_pi_mm = self.critic_2(s_t_detached, act_mm, torch.zeros_like(action_it_hist))
        q_pi_mm = torch.min(q1_pi_mm, q2_pi_mm)

        actor_mm_loss = (self.alpha_0 * log_prob_mm - q_pi_mm).mean()
        self.actor_mm_opt.zero_grad()
        actor_mm_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_mm.parameters(), self.G_max)
        self.actor_mm_opt.step()

        for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
            target_param.data.copy_(TAU * target_param.data + (1 - TAU) * param.data)
        for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
            target_param.data.copy_(TAU * target_param.data + (1 - TAU) * param.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_mm_loss": actor_mm_loss.item(),
            "actor_it_loss": 0.0,
            "dib_loss": 0.0
        }


# ======================================================================
# 4. 高频做市限极订单簿模拟环境 (HFMM LOB Environment)
# ======================================================================
class HFMMEnvironment:
    def __init__(self, lob_data, tick_size=0.01, kappa=0.0003, seq_len=5, max_inventory=MAX_INVENTORY, action_scale=1.0,
                 normalize_reward=False, dataset_type="", initial_capital=10000.0):
        self.lob_data = lob_data.copy().astype(np.float32)
        self.initial_capital = initial_capital
        self.dataset_type = dataset_type.lower()

        self.tick_size = tick_size
        self.kappa = kappa
        self.seq_len = seq_len
        self.max_inventory = max_inventory
        self.action_scale = action_scale
        self.normalize_reward = normalize_reward

        if "binance" in self.dataset_type:
            self.maker_fee_rate = 0.00020
        elif "a_share" in self.dataset_type:
            self.maker_fee_rate = 0.00030
        elif "fi2010" in self.dataset_type:
            self.maker_fee_rate = 0.00010
        else:
            self.maker_fee_rate = 0.00005

        self.reset()

    def reset(self):
        self.current_tick = max(self.seq_len, 16)
        self.cash = self.initial_capital
        self.inventory = 0.0
        self.trade_count = 0
        self.mid_history = []
        self.max_portfolio_value = self.initial_capital
        
        self.last_my_ask = None
        self.last_my_bid = None
        
        return self._get_state()

    def _get_state(self):
        t = self.current_tick

        indices = [
            max(0, t - 16),
            max(0, t - 8),
            max(0, t - 4),
            max(0, t - 2),
            max(0, t - 1)
        ]
        state = self.lob_data[indices].copy().astype(np.float32)

        mid_prices = (state[:, 0] + state[:, 2]) / 2.0
        jump_idx = -1
        for i in range(1, len(mid_prices)):
            prev_mid = mid_prices[i - 1]
            if prev_mid > 1e-6 and (abs(mid_prices[i] - prev_mid) / prev_mid) > 0.015:
                jump_idx = i
                break

        if jump_idx != -1:
            for i in range(jump_idx):
                state[i] = state[jump_idx]

        state_normalized = state.copy().astype(np.float32)

        p1 = state[:, 0]
        p2 = state[:, 2]
        mid = (p1 + p2) / 2.0
        _mid_scalar = float(mid.mean()) if hasattr(mid, 'mean') else float(mid)
        _price_scale = self.tick_size * max(1.0, _mid_scalar / 500.0)

        for j in range(10):
            state_normalized[:, j * 4] = (state_normalized[:, j * 4] - mid) / _price_scale
            state_normalized[:, j * 4 + 2] = (state_normalized[:, j * 4 + 2] - mid) / _price_scale

            vol_ask = np.clip(state[:, j * 4 + 1], 0.0, None)
            vol_bid = np.clip(state[:, j * 4 + 3], 0.0, None)
            state_normalized[:, j * 4 + 1] = np.log1p(vol_ask)
            state_normalized[:, j * 4 + 3] = np.log1p(vol_bid)

        norm_inv_val = self.inventory / self.max_inventory
        inv_col = np.full((self.seq_len, 1), norm_inv_val, dtype=np.float32)
        state_with_inv = np.hstack([state_normalized, inv_col])

        if np.isnan(state_with_inv).any() or np.isinf(state_with_inv).any():
            state_with_inv = np.nan_to_num(state_with_inv, nan=0.0, posinf=0.0, neginf=0.0)

        return torch.tensor(state_with_inv, dtype=torch.float32).unsqueeze(0).to(device)

    def step(self, action_mm, action_it, adv_decay=1.0):
        p1 = self.lob_data[self.current_tick, 0]
        p2 = self.lob_data[self.current_tick, 2]
        ask_1 = max(p1, p2)
        bid_1 = min(p1, p2)
        mid_price = (ask_1 + bid_1) / 2.0

        self.mid_history.append(mid_price)
        if len(self.mid_history) > 100:
            self.mid_history.pop(0)

        if len(self.mid_history) < 2:
            vol = 0.0001
        else:
            log_returns = np.diff(np.log(self.mid_history))
            vol = np.std(log_returns) if np.std(log_returns) > 1e-6 else 0.0001

        act_mm_0 = action_mm[0, 0].item()
        act_mm_1 = action_mm[0, 1].item()
        a_it_val = action_it[0, 0].item()

        if np.isnan(act_mm_0) or np.isinf(act_mm_0): act_mm_0 = 0.0
        if np.isnan(act_mm_1) or np.isinf(act_mm_1): act_mm_1 = 0.0
        if np.isnan(a_it_val) or np.isinf(a_it_val): a_it_val = 0.0

        inv_ratio = self.inventory / self.max_inventory
        skew_ticks = -3.0 * inv_ratio

        if inv_ratio > 0.90:
            offset_bid_ticks = 999.0
            offset_ask_ticks = 1.0
        elif inv_ratio < -0.90:
            offset_ask_ticks = 999.0
            offset_bid_ticks = 1.0
        else:
            offset_ask_ticks = abs(act_mm_0) * self.action_scale + skew_ticks
            offset_bid_ticks = abs(act_mm_1) * self.action_scale - skew_ticks

        offset_ask_ticks = max(0.0, offset_ask_ticks)
        offset_bid_ticks = max(0.0, offset_bid_ticks)

        vol_threshold = 0.00025 if "binance" in self.dataset_type else 0.0005
        gate_multiplier = 1.0
        if vol < vol_threshold:
            gate_multiplier = 1.0 + min(2.5, (vol_threshold - vol) / (vol + 1e-8))

        offset_ask_ticks = offset_ask_ticks * gate_multiplier
        offset_bid_ticks = offset_bid_ticks * gate_multiplier

        my_ask_price = ask_1 + round(offset_ask_ticks) * self.tick_size
        my_bid_price = bid_1 - round(offset_bid_ticks) * self.tick_size

        my_ask_price = max(my_ask_price, bid_1 + self.tick_size)
        my_bid_price = min(my_bid_price, ask_1 - self.tick_size)

        my_ask_price = max(1e-5, my_ask_price)
        my_bid_price = max(1e-5, my_bid_price)

        quoted_spread_ticks = round((my_ask_price - my_bid_price) / self.tick_size)
        dmm_penalty = 0.0
        if quoted_spread_ticks > 6.0:
            dmm_penalty = 0.08 * (quoted_spread_ticks - 6.0)

        cancel_penalty = 0.0
        if self.last_my_ask is not None:
            if abs(my_ask_price - self.last_my_ask) > self.tick_size + 1e-6:
                cancel_penalty += 0.015  
            if abs(my_bid_price - self.last_my_bid) > self.tick_size + 1e-6:
                cancel_penalty += 0.015  
        
        self.last_my_ask = my_ask_price
        self.last_my_bid = my_bid_price

        next_tick = self.current_tick + 1
        if next_tick >= len(self.lob_data):
            next_tick = self.current_tick

        next_p1 = self.lob_data[next_tick, 0]
        next_p2 = self.lob_data[next_tick, 2]
        next_ask = max(next_p1, next_p2)
        next_bid = min(next_p1, next_p2)

        queue_decay_ask = 0.50 if offset_ask_ticks <= 1.0 else 0.85
        queue_decay_bid = 0.50 if offset_bid_ticks <= 1.0 else 0.85
        if next_ask > my_ask_price:
            fill_ask = (np.random.rand() < queue_decay_ask)
        elif next_ask == my_ask_price:
            fill_ask = (np.random.rand() < 0.08)  
        else:
            fill_ask = False

        if next_bid < my_bid_price:
            fill_bid = (np.random.rand() < queue_decay_bid)
        elif next_bid == my_bid_price:
            fill_bid = (np.random.rand() < 0.08)
        else:
            fill_bid = False

        fill_ask_adv = False
        if a_it_val > 0.1:
            dist_ask_ticks = max(0.0, (my_ask_price - ask_1) / self.tick_size)
            prob_adv = a_it_val * np.exp(-0.5 * dist_ask_ticks) * adv_decay
            fill_ask_adv = (np.random.rand() < prob_adv)

        fill_bid_adv = False
        if a_it_val < -0.1:
            dist_bid_ticks = max(0.0, (bid_1 - my_bid_price) / self.tick_size)
            prob_adv = abs(a_it_val) * np.exp(-0.5 * dist_bid_ticks) * adv_decay
            fill_bid_adv = (np.random.rand() < prob_adv)

        fill_ask = fill_ask or fill_ask_adv
        fill_bid = fill_bid or fill_bid_adv

        fill_ask = fill_ask and (self.inventory > -self.max_inventory)
        fill_bid = fill_bid and (self.inventory < self.max_inventory)

        prev_portfolio_value = self.cash + self.inventory * mid_price

        executed = False
        if fill_ask:
            self.inventory -= 1.0
            exec_price = my_ask_price
            fee = exec_price * self.maker_fee_rate
            self.cash += (exec_price - fee)
            executed = True
        if fill_bid:
            self.inventory += 1.0
            exec_price = my_bid_price
            fee = exec_price * self.maker_fee_rate
            self.cash -= (exec_price + fee)
            executed = True

        if executed:
            self.trade_count += 1

        norm_inv = self.inventory / self.max_inventory
        if 'fi2010' in self.dataset_type:
            r_penalty = -5.0 * (norm_inv ** 2)
        else:
            r_penalty = -5.0 * (norm_inv ** 2)

        if 'a_share' in self.dataset_type:
            carrying_cost = -0.20 * abs(self.inventory)
        elif 'fi2010' in self.dataset_type:
            carrying_cost = -0.0005 * abs(self.inventory)
        elif 'binance' in self.dataset_type:
            carrying_cost = -0.001 * abs(self.inventory)
        else:
            carrying_cost = -0.02 * abs(self.inventory)

        liquidation_fee = 0.0
        if abs(self.inventory) >= self.max_inventory * 0.95:
            liquidation_fee = -0.5

        self.current_tick += 1

        curr_p1 = self.lob_data[self.current_tick, 0]
        curr_p2 = self.lob_data[self.current_tick, 2]
        curr_ask = max(curr_p1, curr_p2)
        curr_bid = min(curr_p1, curr_p2)
        curr_mid = (curr_ask + curr_bid) / 2.0

        is_terminal = (self.current_tick >= len(self.lob_data) - 2)
        is_bankrupt = (self.cash + self.inventory * curr_mid <= 0.05 * self.initial_capital)
        done = is_terminal or is_bankrupt

        if done and is_terminal and abs(self.inventory) > 0:
            if self.inventory > 0:
                liquidation_value = self.cash + self.inventory * curr_bid
            else:
                liquidation_value = self.cash + self.inventory * curr_ask

            self.inventory = 0.0
            self.cash = liquidation_value
            current_portfolio_value = liquidation_value
        else:
            current_portfolio_value = self.cash + self.inventory * curr_mid

        raw_r_wealth = current_portfolio_value - prev_portfolio_value

        if 'binance' in self.dataset_type:
            r_wealth = (raw_r_wealth / (prev_portfolio_value + 1e-12)) * 500.0
        elif 'fi2010' in self.dataset_type:
            r_wealth = (raw_r_wealth / (prev_portfolio_value + 1e-12)) * 100.0
        else:
            r_wealth = (raw_r_wealth / (prev_portfolio_value + 1e-12)) * 100.0
        r_wealth = np.clip(r_wealth, -3.0, 3.0)

        if is_bankrupt:
            liquidation_fee -= 10.0

        if current_portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = current_portfolio_value

        drawdown = (self.max_portfolio_value - current_portfolio_value) / (self.max_portfolio_value + 1e-12)
        r_drawdown_penalty = -8.0 * (drawdown ** 2)

        r_over_trading_penalty = 0.0
        if vol < vol_threshold and executed:
            r_over_trading_penalty = -0.15 * (1.0 - (vol / (vol_threshold + 1e-8)))

        total_raw_reward = (
            r_wealth + 
            r_penalty + 
            carrying_cost +        
            liquidation_fee + 
            r_drawdown_penalty + 
            r_over_trading_penalty - 
            cancel_penalty -       
            dmm_penalty            
        )

        reward_scaled = np.clip(total_raw_reward, -3.0, 3.0)
        reward = torch.tensor([[reward_scaled]], dtype=torch.float32).to(device)

        next_state = self._get_state()

        return next_state, reward, done


# ====================== 5. 夏普计算函数 ======================
def calculate_real_daily_sharpe(portfolio_history, dataset_type, bar_size=2000,
                                trade_count=0, initial_capital=10000.0):
    if trade_count < 10:
        return 0.0

    nav_array = np.array(portfolio_history)
    total_ticks = len(nav_array)

    if total_ticks >= 200:
        bar_size = max(5, total_ticks // 100)
    else:
        bar_size = 2

    if total_ticks < bar_size * 2:
        return 0.0

    sampled_nav = nav_array[::bar_size]
    sampled_nav_safe = np.where(sampled_nav[:-1] > 1e-8, sampled_nav[:-1], 1e-8)
    delta_nav_raw = (sampled_nav[1:] - sampled_nav[:-1]) / sampled_nav_safe

    if len(delta_nav_raw) < 5:
        return 0.0

    mean_pnl = np.mean(delta_nav_raw)
    variance = np.var(delta_nav_raw, ddof=1)

    if variance <= 1e-12:
        return 0.0

    std_pnl = np.sqrt(variance) + 1e-8

    if "binance" in dataset_type.lower():
        ticks_per_day = 20000.0
    else:
        ticks_per_day = 4800.0

    bars_per_day = ticks_per_day / bar_size
    scale_factor_daily = np.sqrt(bars_per_day)

    raw_daily_sharpe = (mean_pnl / std_pnl) * scale_factor_daily

    sample_size = len(delta_nav_raw)
    small_sample_correction = np.sqrt(sample_size / (sample_size + 15.0))

    final_daily_sharpe = raw_daily_sharpe * small_sample_correction
    final_daily_sharpe = float(np.clip(final_daily_sharpe, -100.0, 100.0))

    return final_daily_sharpe


# ======================================================================
# 6. 通用 LOB 数据加载与“数据防崩卫士”对齐接口
# ======================================================================
def load_lob_data(filepath, dataset_type="fi2010", max_ticks=None, is_real_target=True):
    if filepath is not None and os.path.exists(filepath):
        print(f"\n正在加载【{dataset_type.upper()}】高频 LOB 数据: {filepath} ...", flush=True)

        if dataset_type == "a_share":
            cols = []
            for i in range(1, 11):
                cols.extend([f'AskPrice{i}', f'AskVolume{i}', f'BidPrice{i}', f'BidVolume{i}'])

            if max_ticks is not None:
                df = pd.read_csv(filepath, usecols=cols, nrows=max_ticks + 100)
            else:
                df = pd.read_csv(filepath, usecols=cols)

            raw_data_t = df[cols].values
            print("   👉 成功完成 A股 专属数据列格式对齐。", flush=True)

        elif dataset_type == "lob_bench" and os.path.isdir(filepath):
            all_csvs = glob.glob(os.path.join(filepath, "**", "*.csv"), recursive=True)
            orderbook_files = [f for f in all_csvs if "orderbook" in os.path.basename(f).lower()]

            if len(orderbook_files) == 0:
                raise FileNotFoundError(
                    f"❌ [数据验证失败] 在目录 {filepath} 下无法检索到任何做市订单簿 (orderbook) 数据文件。"
                )

            matched_files = sorted(orderbook_files)
            print(f"   👉 成功匹配并定位到 {len(matched_files)} 个 LOB-Bench 订单簿分卷，合并加载中...", flush=True)
            dfs = []
            for f in matched_files[:15]:
                dfs.append(pd.read_csv(f, header=None))
            df = pd.concat(dfs, ignore_index=True)

            raw_values = df.values[:, :40]
            reordered_cols = []
            for i in range(10):
                reordered_cols.extend([i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3])
            raw_data_t = raw_values[:, reordered_cols]
            print("   👉 成功完成 LOB-Bench 数据列格式对齐。", flush=True)

        elif filepath.endswith('.pt'):
            raw_data_t = torch.load(filepath, map_location='cpu', weights_only=False)
            if isinstance(raw_data_t, dict):
                found_key = None
                for k in ['train', 'val', 'test', 'dataset', 'data', 'x', 'features', 'lob', 'inputs', 'train_x',
                          'test_x', 'val_x']:
                    if k in raw_data_t:
                        found_key = k
                        break
                if found_key is not None:
                    raw_data_t = raw_data_t[found_key]
                else:
                    tensors = [v for v in raw_data_t.values() if torch.is_tensor(v) or isinstance(v, np.ndarray)]
                    if len(tensors) == 1:
                        raw_data_t = tensors[0]
                    else:
                        raise ValueError("无法从该字典数据中识别有效特征矩阵")

            elif isinstance(raw_data_t, tuple) or isinstance(raw_data_t, list):
                raw_data_t = raw_data_t[0]

            if torch.is_tensor(raw_data_t):
                raw_data_t = raw_data_t.numpy()

            if len(raw_data_t.shape) == 3:
                if raw_data_t.shape[2] == 40:
                    raw_data_t = raw_data_t[:, -1, :]
                elif raw_data_t.shape[1] == 40:
                    raw_data_t = raw_data_t[:, :, -1]

            if len(raw_data_t.shape) == 2:
                if raw_data_t.shape[0] == 40 and raw_data_t.shape[1] > 40:
                    raw_data_t = raw_data_t.T
            print("   👉 成功从本地 PyTorch .pt 预处理张量文件加载并转换。", flush=True)

        elif dataset_type == "fi2010":
            raw_data = np.loadtxt(filepath)
            if raw_data.shape[0] == 149:
                raw_data_t = raw_data.T
            else:
                raw_data_t = raw_data

        elif dataset_type == "binance":
            df = pd.read_csv(filepath)
            lob_cols = df.iloc[:, 2:42].values
            reordered_data = np.zeros_like(lob_cols)
            for i in range(10):
                reordered_data[:, i * 4] = lob_cols[:, 20 + i * 2]
                reordered_data[:, i * 4 + 1] = lob_cols[:, 20 + i * 2 + 1]
                reordered_data[:, i * 4 + 2] = lob_cols[:, i * 2]
                reordered_data[:, i * 4 + 3] = lob_cols[:, i * 2 + 1]
            raw_data_t = reordered_data
            print("   👉 成功完成 Binance 数据列格式对齐（按 LOBSTER 标准）。", flush=True)
        else:
            raise ValueError(f"未知的数据集类型: {dataset_type}")

        raw_data_t = np.asarray(raw_data_t, dtype=np.float32)

        is_normalized = (raw_data_t[:, [0, 2]] < 0.0).any() or (np.abs(raw_data_t[:, [0, 2]].mean()) < 5.0)

        vol_cols = [i * 4 + 1 for i in range(10)] + [i * 4 + 3 for i in range(10)]
        for col in vol_cols:
            bad_mask = np.isnan(raw_data_t[:, col]) | np.isinf(raw_data_t[:, col])
            if bad_mask.any():
                temp_series = pd.Series(raw_data_t[:, col])
                temp_series[bad_mask] = np.nan
                temp_series = temp_series.ffill().bfill().fillna(0.0)
                raw_data_t[:, col] = temp_series.values
            raw_data_t[:, col] = np.clip(raw_data_t[:, col], 0.0, None)

        if is_normalized:
            print("   ⚠️ [数据卫士] 检测到输入数据已完成归一化。", flush=True)
            print("   🔄 启动【自适应绝对特征重构】，将其归至物理空间...", flush=True)

            price_cols = [i * 4 for i in range(10)] + [i * 4 + 2 for i in range(10)]
            for col in price_cols:
                col_std = raw_data_t[:, col].std()
                if col_std > 1e-6:
                    raw_data_t[:, col] = 100.0 + ((raw_data_t[:, col] - raw_data_t[:, col].mean()) / col_std) * 5.0
                else:
                    raw_data_t[:, col] = 100.0

            vol_cols_norm = [i * 4 + 1 for i in range(10)] + [i * 4 + 3 for i in range(10)]
            for col in vol_cols_norm:
                col_std = raw_data_t[:, col].std()
                if col_std > 1e-6:
                    raw_data_t[:, col] = 100.0 + ((raw_data_t[:, col] - raw_data_t[:, col].mean()) / col_std) * 30.0
                else:
                    raw_data_t[:, col] = 100.0
                raw_data_t[:, col] = np.clip(raw_data_t[:, col], 1.0, None)

            print("   👉 [绝对特征重构成功] 所有价格列已无损映射，挂单量已映射回物理空间。", flush=True)
        else:
            for col_idx in [0, 2]:
                bad_mask = (raw_data_t[:, col_idx] <= 1e-6) | np.isnan(raw_data_t[:, col_idx]) | np.isinf(
                    raw_data_t[:, col_idx]) | (raw_data_t[:, col_idx] > 9_000_000_000)
                if bad_mask.any():
                    print(
                        f"   ⚠️ [数据卫士] 检测到第 {col_idx} 列存在 {bad_mask.sum()} 个异常价格 Tick，执行前向填充...", flush=True)
                    temp_series = pd.Series(raw_data_t[:, col_idx])
                    temp_series[bad_mask] = np.nan
                    temp_series = temp_series.ffill().bfill().fillna(1.0)
                    raw_data_t[:, col_idx] = temp_series.values

            if raw_data_t[:, 0].mean() > 10000.0 and dataset_type in ["lob_bench", "trades_lob"]:
                print("   ⚠️ [数据卫士] 检测到价格时序量级过大（均价 > 10,000），自动还原 10,000 倍标的价格。", flush=True)
                for col_idx in range(40):
                    if col_idx % 2 == 0:
                        raw_data_t[:, col_idx] /= 10000.0
            else:
                if raw_data_t[:, 0].mean() > 10000.0:
                    print(
                        f"   ℹ [数据卫士] 检测到高价值物理标的价格 ({raw_data_t[:, 0].mean():.2f})，确认为真实交易空间，跳过价格还原。", flush=True)

            for col_idx in range(40):
                if col_idx % 2 == 1:
                    raw_data_t[:, col_idx] = np.clip(raw_data_t[:, col_idx], 0.0, None)

        if max_ticks is not None and len(raw_data_t) > max_ticks:
            raw_data_t = raw_data_t[:max_ticks]
            print(f"   🚀 已启用极速切片！只截取前 {max_ticks} 个交易 Tick。", flush=True)
        else:
            print(f"   🚀 已启用全量数据集！载入 {len(raw_data_t)} 个交易 Ticks 运行。", flush=True)

        return raw_data_t[:, :40]
    else:
        raise FileNotFoundError(f"未在当前目录下找到所需的 {filepath} 文件，请检查该文件是否存在！")


# ======================================================================
# 7. 一键顺序自动联调/参数化运行主程序
# ======================================================================
def main():
    global device

    parser = argparse.ArgumentParser(description="DIB-MASAC Multi-Market Engine")
    parser.add_argument(
        "--dataset",
        type=str,
        default="a_share_sz000001",
        help="指定运行的目标市场，支持逗号分隔多个或'all'。"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="指定使用的 GPU 卡号"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="our",
        help="运行模式。支持单个、英文逗号分隔的多个模式 or 'all'"
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="指定运行的学术随机种子，用英文逗号隔开"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="开启极速调参模式"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="忽略已有的历史运行和模型文件，强制重新训练和跑数"
    )
    parser.add_argument("--lr", type=float, default=None, help="覆盖全局学习率LR")
    parser.add_argument("--kl_coef", type=float, default=None, help="覆盖DIB的KL系数")
    parser.add_argument("--alpha", type=float, default=None, help="覆盖SAC熵系数ALPHA_0")
    args = parser.parse_args()

    # 命令行覆盖全局超参数
    global LR, KL_COEF, ALPHA_0
    if args.lr is not None: LR = args.lr
    if args.kl_coef is not None: KL_COEF = args.kl_coef
    if args.alpha is not None: ALPHA_0 = args.alpha

    # 解析种子列表
    seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip().isdigit()]

    if args.fast:
        seed_list = [seed_list[0]] if len(seed_list) > 0 else [0]
        current_max_ticks = 20000
        _DS_TICK_LIMITS = {
            "binance": 20000,
            "a_share": 20000,
            "fi2010": 20000,
            "trades_lob": 20000,
            "lob_bench": None,
        }  
        print(f"⚡ [极速调参模式激活] 数据 Tick 限制为 {current_max_ticks} | 强制仅运行单种子: {seed_list}", flush=True)
    else:
        # ✅ 按dataset类型限制max_ticks，防止大数据集跑死
        _DS_TICK_LIMITS = {
            "binance":    50000,
            "a_share":    40000,
            "fi2010":     50000,
            "trades_lob": 50000,
            "lob_bench":  None,   # 本来就小，不截断
        }
        current_max_ticks = MAX_TICKS  # 默认值

    all_modes = ["our", "our_v2", "our_v3", "ablation_no_dib", "ablation_no_it", "as", "ho_stoll", "ppo", "sac", "hrl", "mappo", "deeplob", "timesfm", "moment", "timer_xl"]
    if args.mode.lower() == "all":
        modes_to_test = all_modes
    else:
        modes_to_test = [m.strip().lower() for m in args.mode.split(",") if m.strip().lower() in all_modes]
        if not modes_to_test:
            modes_to_test = ["our"]

    if torch.cuda.is_available():
        gpu_id = min(max(0, args.gpu), torch.cuda.device_count() - 1)
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
    print(f"当前激活设备: {device} | 评估模型队列: {modes_to_test} | 学术种子队列: {seed_list}", flush=True)

    all_datasets = [
        "fi2010",
        "lob_bench_goog_real",
        "lob_bench_goog_synth",  
        "lob_bench_intc_real",
        "lob_bench_intc_synth",  
        "binance_low_vol", "binance_high_vol",
        "trades_lob_tsla", "trades_lob_intc",
        "a_share_sz000001", "a_share_sz000651", "a_share_sz002415", "a_share_sz300147"
    ]

    target_dataset = args.dataset

    if args.dataset == "all":
        datasets_to_test = all_datasets
    elif "," in args.dataset:
        datasets_to_test = [d.strip() for d in args.dataset.split(",") if d.strip() in all_datasets]
        print(f"🚀 多市场模式：{[d.upper() for d in datasets_to_test]}", flush=True)
    else:
        datasets_to_test = [target_dataset]
        print(f"🚀 专项训练和评估市场：【{target_dataset.upper()}】", flush=True)

    dataset_summary_stats = {d: {m: [] for m in modes_to_test} for d in datasets_to_test}
    test_results = {}

    print("\n" + "=" * 65, flush=True)
    print(f"🚀 【学术对齐版实验系统】启动 | 参评模型: {[m.upper() for m in modes_to_test]}", flush=True)
    print("=" * 65, flush=True)

    for DATASET_TYPE in datasets_to_test:
        print("\n" + "#" * 60, flush=True)
        print(f"🎬 [市场数据准备] 载入数据集: {DATASET_TYPE.upper()}", flush=True)
        print("#" * 60, flush=True)

        norm_rew = True

        try:
            current_tick_size = TICK_SIZES.get(DATASET_TYPE, 0.01)

            if "a_share" in DATASET_TYPE:
                ticker = DATASET_TYPE.split("_")[-1]
                matched_file = find_file_path(f"*{ticker}*.csv")
                TRAIN_FILE = matched_file if matched_file else f"{ticker}-level10_processed.csv"
                VAL_FILE = TRAIN_FILE
                ACTION_SCALE = 1.0
                LOADER_TYPE = "a_share"

            elif "trades_lob" in DATASET_TYPE:
                ticker = DATASET_TYPE.split("_")[-1]
                matched_file = find_file_path(f"*lob*bench*{ticker}*real*.pt") or find_file_path(f"*{ticker}*.pt")
                TRAIN_FILE = matched_file if matched_file else f"lob_bench_{ticker}_real.pt"
                VAL_FILE = TRAIN_FILE
                ACTION_SCALE = 1.0
                LOADER_TYPE = "trades_lob"

            elif DATASET_TYPE == "fi2010":
                matched_file = find_file_path("fi-2010.pt") or find_file_path("*CF_7.txt")
                TRAIN_FILE = matched_file if matched_file else "fi_2010_toy.pt"
                VAL_FILE = TRAIN_FILE
                ACTION_SCALE = 1.0
                LOADER_TYPE = "fi2010"

            elif DATASET_TYPE == "binance_low_vol":
                matched_file = find_file_path("binance_low_vol.pt") or find_file_path("binance_low_vol_toy.pt")
                TRAIN_FILE = matched_file if matched_file else "binance_low_vol_toy.pt"
                VAL_FILE = TRAIN_FILE
                ACTION_SCALE = 1.0
                LOADER_TYPE = "binance"

            elif DATASET_TYPE == "binance_high_vol":
                matched_file = find_file_path("binance_high_vol.pt") or find_file_path("binance_high_vol_toy.pt")
                TRAIN_FILE = matched_file if matched_file else "binance_high_vol_toy.pt"
                VAL_FILE = TRAIN_FILE
                ACTION_SCALE = 1.0
                LOADER_TYPE = "binance"

            elif "lob_bench" in DATASET_TYPE:
                asset_dir = "GOOG" if "goog" in DATASET_TYPE else "INTC"
                base_dir = find_file_path("lob_bench_data")
                if base_dir and os.path.exists(os.path.join(base_dir, asset_dir)):
                    TRAIN_FILE = os.path.join(base_dir, asset_dir)
                else:
                    suffix = "synth" if "synth" in DATASET_TYPE else "real"
                    TRAIN_FILE = (
                            find_file_path(f"*lob_bench_{asset_dir.lower()}*{suffix}*.pt") or
                            find_file_path(f"*{asset_dir.lower()}_{suffix}.pt") or
                            find_file_path(f"*{asset_dir.lower()}*.pt") or
                            f"{DATASET_TYPE}_toy.pt"
                    )
                VAL_FILE = TRAIN_FILE
                # ✅ INTC价格337000量级，tick=100，用0.5让报价贴近市场提高成交率
                ACTION_SCALE = 1.0
                LOADER_TYPE = "lob_bench"
            else:
                raise ValueError(f"未知数据集类型: {DATASET_TYPE}")

            config = {
                "max_inventory": 50.0,
                "action_scale": ACTION_SCALE,
            }

            is_real = "real" in DATASET_TYPE
            # ✅ 按LOADER_TYPE动态覆盖max_ticks
            _loader_key = LOADER_TYPE if LOADER_TYPE in _DS_TICK_LIMITS else "lob_bench"
            current_max_ticks = _DS_TICK_LIMITS[_loader_key]
            train_lob = load_lob_data(filepath=TRAIN_FILE, dataset_type=LOADER_TYPE, max_ticks=current_max_ticks,
                                      is_real_target=is_real)

            if TRAIN_FILE == VAL_FILE:
                split_idx = int(len(train_lob) * 0.8)
                val_lob = train_lob[split_idx:]
                train_lob = train_lob[:split_idx]
                print("   ⚠️ 已自动执行 Train/Val 单向切分 (80% 训练，20% 样本外验证)。", flush=True)
            else:
                val_lob = load_lob_data(filepath=VAL_FILE, dataset_type=LOADER_TYPE, max_ticks=current_max_ticks,
                                        is_real_target=is_real)

            print(f"数据切分完成 -> 训练集 Ticks: {len(train_lob)} | 样本外测试集 (Val) Ticks: {len(val_lob)}", flush=True)

            # our_v2：基于训练集自动计算自适应动作缩放
            spread_vals = np.abs(train_lob[:, 0] - train_lob[:, 2])
            mean_spread = float(np.median(spread_vals[spread_vals > 1e-6]))
            GLOBAL_K = 0.5
            adaptive_action_scale = float(np.clip(GLOBAL_K * mean_spread / current_tick_size, 0.5, 15.0))
            print(f"   [自适应缩放] mean_spread={mean_spread:.6f}, tick={current_tick_size}, adaptive_scale={adaptive_action_scale:.4f}", flush=True)

            if args.fast:
                epochs_to_run = 2
            elif EVALUATE_MODE:
                epochs_to_run = 20  # ✅ 统一12epoch，截断后长度判断已无意义
            else:
                epochs_to_run = 3

            starting_p1 = train_lob[0, 0]
            starting_p2 = train_lob[0, 2]
            starting_mid = (max(starting_p1, starting_p2) + min(starting_p1, starting_p2)) / 2.0

            if 'fi2010' in DATASET_TYPE:
                adaptive_capital = 500.0
                ACTION_SCALE = 0.5
            elif starting_mid <= 1.0:
                adaptive_capital = 1000.0
            else:
                adaptive_capital = starting_mid * config["max_inventory"] * 3.0

            config["action_scale"] = float(ACTION_SCALE)

            if LOADER_TYPE == "binance":
                current_max_gate = 0.05      
                adaptive_alpha_val = 0.02    
                current_as_kappa = 500.0     # BTC tick_size=0.1，adaptive_kappa=5000，让spread贴近市场
                current_delta_scale = 0.3
            elif LOADER_TYPE == "a_share":
                current_max_gate = 0.05      
                adaptive_alpha_val = 0.02    
                current_as_kappa = 1.5
                current_delta_scale = 0.3    # action_scale=10，0.3×10=3，等效其他市场的3×1
                if "sz002415" in DATASET_TYPE:
                    current_max_gate = 0.05   # 保持默认gate
                    adaptive_alpha_val = 0.05  # 提高熵正则，让策略更保守
                    current_as_kappa = 0.5     # 降低kappa让AS prior给更大spread
            elif LOADER_TYPE == "fi2010":
                current_max_gate = 0.50      
                adaptive_alpha_val = 0.10    
                current_as_kappa = 0.5       # fi2010 tick_size=0.01，adaptive_kappa=50，给AS prior有效指导
                current_delta_scale = 1.0
            elif "intc" in DATASET_TYPE:
                # ✅ INTC波动幅度小（mid在2-3tick范围内波动），降低gate和kappa让报价贴近市场
                current_max_gate = 0.15
                adaptive_alpha_val = ALPHA_0
                current_as_kappa = 5.0
                current_delta_scale = 3.0
            else:
                current_max_gate = 3.0       
                adaptive_alpha_val = ALPHA_0 
                current_as_kappa = 1.5
                if "goog" in DATASET_TYPE:
                    adaptive_alpha_val = 0.05  # goog策略方差大，保守化
                current_delta_scale = 3.0

            adaptive_max_offset = 1.0 if "intc" in DATASET_TYPE else 3.0

            as_prior_generator = AvellanedaStoikovAgent(gamma=0.1, kappa=current_as_kappa)

            for mode in modes_to_test:
                print(f"\n⚡ [模型切换] 启动模型评估: 【{mode.upper()}】", flush=True)
                
                for SEED_VAL in seed_list:
                    progress_log_file = f"aaai_experiment_progress_seed{SEED_VAL}.csv"
                    # ✅ 断点续跑：检查该(dataset, mode, seed)是否已完成，若是则跳过
                    _already_done = False
                    if os.path.isfile(progress_log_file):
                        import csv as _csv
                        with open(progress_log_file, 'r', encoding='utf-8') as _f:
                            for _row in _csv.DictReader(_f):
                                if (_row.get('dataset') == DATASET_TYPE and
                                        _row.get('mode') == mode and
                                        str(_row.get('seed')) == str(SEED_VAL)):
                                    _already_done = True
                                    break
                    if _already_done and not args.force:
                        print(f"⏭ [跳过] DS:{DATASET_TYPE} MODEL:{mode} SEED:{SEED_VAL} 已在CSV中，跳过。", flush=True)
                        continue
                    
                    # 🎯 设定独立保存目录与极高辨识度的文件分区命名，区分“我们的方法”与“基线对比方法”
                    is_our = (mode in ("our", "our_v2", "our_v3"))
                    output_dir = "OUR_PROPOSED_DIB_MASAC" if is_our else "BASELINE_COMPARISONS"
                    os.makedirs(output_dir, exist_ok=True)

                    # 🎯 为避免覆盖以及准确记录时间，引入执行时的时间戳和秒表
                    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    seed_start_time = time.time()

                    final_traj_path = os.path.join(output_dir, f"trajectory_data_{mode}_{DATASET_TYPE}_seed{SEED_VAL}_{run_timestamp}.csv")
                    final_model_path = os.path.join(output_dir, f"best_hfmm_model_{mode}_{DATASET_TYPE}_seed{SEED_VAL}_{run_timestamp}.pth")

                    try:
                        print(f"🌱 [全新激活运行] SEED = {SEED_VAL} 正在完全重新训练/评估...", flush=True)
                        set_seed(SEED_VAL)

                        is_heuristic = mode in ["as", "ho_stoll"]

                        if mode == "our":
                            agent = StabilizedAdversarialMASAC(
                                state_dim=32, 
                                action_dim_mm=2, 
                                action_dim_it=1, 
                                alpha_0=adaptive_alpha_val
                            )
                        elif mode == "our_v2":
                            agent = StabilizedAdversarialMASAC(
                                state_dim=32,
                                action_dim_mm=2,
                                action_dim_it=1,
                                alpha_0=adaptive_alpha_val
                            )
                            config["action_scale"] = adaptive_action_scale
                            print(f"   [our_v2] 使用自适应 action_scale={adaptive_action_scale:.4f}", flush=True)
                        elif mode == "our_v3":
                            agent = StabilizedAdversarialMASAC(
                                state_dim=32,
                                action_dim_mm=2,
                                action_dim_it=1,
                                alpha_0=adaptive_alpha_val
                            )
                            agent.encoder = BiGRUDIBEncoder(latent_dim=32).to(device)
                            agent.encoder_target = BiGRUDIBEncoder(latent_dim=32).to(device)
                            agent.encoder_target.load_state_dict(agent.encoder.state_dict())
                            config["action_scale"] = adaptive_action_scale
                            print(f"   [our_v3] BiGRU编码器 + 自适应 action_scale={adaptive_action_scale:.4f}", flush=True)
                        elif mode == "as":
                            agent = AvellanedaStoikovAgent(gamma=0.1, kappa=current_as_kappa)
                        elif mode == "ho_stoll":
                            agent = HoStollAgent(gamma=0.1)
                        elif mode == "ppo":
                            agent = SinglePPOAgent(state_dim=32, action_dim_mm=2)
                        elif mode == "hrl":
                            agent = HierarchicalRLAgent(state_dim=32, action_dim_mm=2)
                        elif mode == "mappo":
                            agent = StandardMAPPO(state_dim=32, action_dim_mm=2, action_dim_it=1)
                        elif mode in ["sac", "deeplob", "timesfm", "moment", "timer_xl"]:
                            agent = GeneralizedSACBaseline(encoder_type=mode, state_dim=32, action_dim_mm=2)
                        elif mode == "ablation_no_dib":
                            agent = StabilizedAdversarialMASAC(
                                state_dim=32,
                                action_dim_mm=2,
                                action_dim_it=1,
                                alpha_0=adaptive_alpha_val,
                                ablation_mode="no_dib"
                            )
                        elif mode == "ablation_no_it":
                            agent = StabilizedAdversarialMASAC(
                                state_dim=32,
                                action_dim_mm=2,
                                action_dim_it=1,
                                alpha_0=adaptive_alpha_val,
                                ablation_mode="no_it"
                            )
                        else:
                            raise ValueError(f"未知的运行模式: {mode}")

                        if not is_heuristic and agent is not None:
                            for module in agent.modules() if hasattr(agent, "modules") else []:
                                module.to(device)
                            if hasattr(agent, "encoder") and agent.encoder is not None:
                                agent.encoder.to(device)
                            if hasattr(agent, "actor_mm") and agent.actor_mm is not None:
                                agent.actor_mm.to(device)
                            if hasattr(agent, "actor_it") and agent.actor_it is not None:
                                agent.actor_it.to(device)
                            if hasattr(agent, "critic_1") and agent.critic_1 is not None:
                                agent.critic_1.to(device)
                            if hasattr(agent, "critic_2") and agent.critic_2 is not None:
                                agent.critic_2.to(device)
                            if hasattr(agent, "critic") and agent.critic is not None:
                                agent.critic.to(device)
                            if hasattr(agent, "manager") and agent.manager is not None:
                                agent.manager.to(device)
                            if hasattr(agent, "reward_decoder") and agent.reward_decoder is not None:
                                agent.reward_decoder.to(device)

                        if not is_heuristic and agent is not None:
                            print(f"   ℹ️ [权重卫士] 当前模式：{mode.upper()} 已经从全新随机初值启动跑圈，跳过一切预载机制。", flush=True)

                        batch_size = BATCH_SIZE
                        update_every = 30 if args.fast else UPDATE_EVERY

                        buffer = ReplayBuffer(capacity=50000, device=device)
                        best_val_utility = -float('inf')

                        print(f"开始运行 【{mode.upper()}】 | 种子: {SEED_VAL}", flush=True)
                        print(f"   自适应资产价格: {starting_mid:.4f} | 物理报价 Tick Size: {current_tick_size}", flush=True)
                        print(f"   自适应做市宽度 (Action Scale): {config['action_scale']:.2f} Ticks", flush=True)
                        print(f"   自适应准备金 (Capital): {adaptive_capital:.2f}", flush=True)
                        print("-" * 60, flush=True)

                        if is_heuristic:
                            val_env = HFMMEnvironment(
                                val_lob,
                                tick_size=current_tick_size,
                                kappa=0.0003,
                                seq_len=5,
                                max_inventory=config["max_inventory"],
                                action_scale=config["action_scale"],
                                normalize_reward=norm_rew,
                                dataset_type=LOADER_TYPE,
                                initial_capital=adaptive_capital
                            )
                            val_state = val_env.reset()
                            val_done = False

                            portfolio_history = []
                            val_inventories = []

                            while not val_done:
                                p1 = val_env.lob_data[val_env.current_tick - 1, 0]
                                p2 = val_env.lob_data[val_env.current_tick - 1, 2]
                                curr_mid = (max(p1, p2) + min(p1, p2)) / 2.0

                                action_mm_val = agent.get_action(curr_mid, val_env.inventory, current_tick_size, max_offset=adaptive_max_offset)
                                action_it_val = torch.zeros(1, 1).to(device)

                                next_val_state, val_reward, val_done = val_env.step(action_mm_val, action_it_val)
                                val_inventories.append(val_env.inventory)

                                p_val = val_env.cash + val_env.inventory * curr_mid
                                portfolio_history.append(p_val)

                                val_state = next_val_state

                            val_sharpe_real = calculate_real_daily_sharpe(
                                portfolio_history=portfolio_history,
                                dataset_type=DATASET_TYPE,
                                bar_size=2000,
                                trade_count=val_env.trade_count,
                                initial_capital=adaptive_capital
                            )

                            full_nav = np.array(portfolio_history)
                            cum_max = np.maximum.accumulate(full_nav)
                            denominator = np.maximum(cum_max, adaptive_capital)
                            drawdowns = (cum_max - full_nav) / (denominator + 1e-12)
                            val_max_dd = np.clip(np.max(drawdowns), 0.0, 1.0)

                            val_inventories_var = float(np.var(val_inventories))
                            val_utility = val_sharpe_real - 0.05 * val_inventories_var

                            print(
                                f"==> 经典 {mode.upper()} (种子 {SEED_VAL}) 评估结束 | "
                                f"效用 (Utility): {val_utility:.4f} | "
                                f"夏普 (Sharpe): {val_sharpe_real:.4f} | "
                                f"方差 (Variance): {val_inventories_var:.4f} | "
                                f"最大回撤: {val_max_dd * 100:.2f}%",
                                flush=True
                            )

                            df_traj = pd.DataFrame({
                                "step": range(len(portfolio_history)),
                                "portfolio_value": portfolio_history,
                                "inventory": val_inventories
                            })
                            df_traj.to_csv(final_traj_path, index=False)
                            
                            seed_elapsed = time.time() - seed_start_time
                            save_result_to_csv(
                                progress_log_file, DATASET_TYPE, mode, SEED_VAL,
                                val_utility, val_sharpe_real, val_inventories_var, val_max_dd, val_env.trade_count,
                                start_time_str, seed_elapsed, is_our, output_dir
                            )
                            plot_and_save_trajectory(portfolio_history, val_inventories, mode, DATASET_TYPE, SEED_VAL, output_dir, run_timestamp)
                            
                            dataset_summary_stats[DATASET_TYPE][mode].append(val_utility)
                            test_results[f"{DATASET_TYPE}_{mode}_seed{SEED_VAL}"] = f"✅ 学术跑数成功 (最佳 Utility: {val_utility:.4f})"
                            continue

                        # ====================== 神经网络模型跑数循环 ======================
                        best_val_utility = -float('inf')
                        best_portfolio_history = []
                        best_val_inventories = []
                        
                        for epoch in range(1, epochs_to_run + 1):
                            print(f"[TRAIN EPOCH] DS:{DATASET_TYPE} MODEL:{mode} SEED:{SEED_VAL} Epoch:{epoch}/{epochs_to_run} UTIL:{best_val_utility:.4f}", flush=True)
                            env = HFMMEnvironment(
                                train_lob,
                                tick_size=current_tick_size,
                                kappa=0.0003,
                                seq_len=5,
                                max_inventory=config["max_inventory"],
                                action_scale=config["action_scale"],
                                normalize_reward=norm_rew,
                                dataset_type=LOADER_TYPE,
                                initial_capital=adaptive_capital
                            )
                            state = env.reset()
                            done = False
                            epoch_rewards = []
                            step_count = 0
                            loss_dict = {"critic_loss": 0.0, "dib_loss": 0.0}

                            current_adv_decay = max(0.01, 1.0 - (epoch - 1) / (epochs_to_run - 0.999))

                            while not done:
                                with torch.no_grad():
                                    if mode in ("our", "our_v2", "our_v3", "ablation_no_dib", "ablation_no_it"):
                                        s_t, _, _ = agent.encoder(state)
                                        if getattr(agent, 'ablation_mode', None) == 'no_it':
                                            action_it = torch.zeros(1, 1, device=device)
                                        else:
                                            action_it, _ = agent.actor_it(s_t)
                                        s_worker = torch.cat([s_t, action_it], dim=-1)
                                        action_mm_raw, _ = agent.actor_mm(s_worker)

                                        p1 = env.lob_data[env.current_tick, 0]
                                        p2 = env.lob_data[env.current_tick, 2]
                                        mid_val = (p1 + p2) / 2.0
                                        action_as = as_prior_generator.get_action(mid_val, env.inventory, current_tick_size, max_offset=adaptive_max_offset)
                                        
                                        if LOADER_TYPE == 'fi2010':
                                            raw_off = torch.clamp(torch.abs(action_mm_raw[..., :2]) * config['action_scale'] * 2.0, 0.01, 3.0)
                                            inv_bias = float(np.clip(env.inventory / env.max_inventory * 0.5, -0.5, 0.5))
                                            bid_off = torch.clamp(raw_off[..., 0:1] + inv_bias, 0.01, 3.0)
                                            ask_off = torch.clamp(raw_off[..., 1:2] - inv_bias, 0.01, 3.0)
                                            action_mm = torch.cat([bid_off, ask_off], dim=-1)
                                        elif LOADER_TYPE in ['lob_bench', 'trades_lob', 'a_share']:
                                            action_mm = torch.clamp(torch.abs(action_mm_raw[..., :2]) * config['action_scale'] * 2.0, 0.01, 3.0)
                                        else:
                                            if action_as.shape[-1] == 0: action_as = torch.ones(1, 2, device=device) * 1.0
                                            action_mm = blend_action(action_mm_raw, action_as, config["action_scale"], max_gate=current_max_gate, min_val=0.0)
                                        next_state, reward, done = env.step(action_mm, action_it, adv_decay=current_adv_decay)
                                        buffer.push(state, action_mm_raw, action_it, reward, next_state, float(done))

                                    elif mode == "mappo":
                                        s_t = agent.encoder(state)
                                        action_mm, _ = agent.actor_mm(s_t)
                                        action_it, _ = agent.actor_it(s_t)
                                        next_state, reward, done = env.step(action_mm, action_it)
                                        buffer.push(state, action_mm, action_it, reward, next_state, float(done))

                                    elif mode in ["sac", "deeplob", "timesfm", "moment", "timer_xl"]:
                                        s_t = agent.encoder(state)
                                        action_mm, _ = agent.actor_mm(s_t)
                                        action_it = torch.zeros(1, 1, device=device)
                                        next_state, reward, done = env.step(action_mm, action_it)
                                        buffer.push(state, action_mm, action_it, reward, next_state, float(done))

                                    elif mode == "ppo":
                                        s_t = agent.encoder(state)
                                        action_mm, _ = agent.actor_mm(s_t)
                                        action_it = torch.zeros(1, 1, device=device)
                                        next_state, reward, done = env.step(action_mm, action_it)
                                        buffer.push(state, action_mm, action_it, reward, next_state, float(done))

                                    elif mode == "hrl":
                                        s_t = agent.encoder(state)
                                        goal, _ = agent.manager(s_t)
                                        s_worker = torch.cat([s_t, goal], dim=-1)
                                        action_mm, _ = agent.actor_mm(s_worker)
                                        action_it = torch.zeros(1, 1, device=device)
                                        next_state, reward, done = env.step(action_mm, action_it)
                                        buffer.push(state, action_mm, action_it, reward, next_state, float(done))

                                state = next_state
                                epoch_rewards.append(reward.item())
                                step_count += 1

                                if step_count % update_every == 0 and len(buffer) >= batch_size:
                                    b_state, b_act_mm, b_act_it, b_reward, b_next_state, b_done = buffer.sample(batch_size)
                                    loss_dict = agent.train_step(
                                        L_t=b_state,
                                        action_mm_hist=b_act_mm,
                                        action_it_hist=b_act_it,
                                        r_t=b_reward,
                                        L_next=b_next_state,
                                        done=b_done
                                    )

                                if step_count % 5000 == 0:
                                    print(f"Epoch {epoch:2d} | Step {step_count:6d}/{len(train_lob)} | "
                                          f"Q_loss: {loss_dict['critic_loss']:.4f} | "
                                          f"Avg_Reward: {np.mean(epoch_rewards[-100:]):.6f}", flush=True)

                            # 验证回测
                            val_env = HFMMEnvironment(
                                val_lob,
                                tick_size=current_tick_size,
                                kappa=0.0003,
                                seq_len=5,
                                max_inventory=config["max_inventory"],
                                action_scale=config["action_scale"],
                                normalize_reward=norm_rew,
                                dataset_type=LOADER_TYPE,
                                initial_capital=adaptive_capital
                            )
                            val_state = val_env.reset()
                            val_done = False

                            portfolio_history = []
                            val_inventories = []

                            step_cnt = 0
                            total_val_ticks = len(val_lob) 

                            while not val_done:
                                step_cnt += 1
                                with torch.no_grad():
                                    action_it_val = torch.zeros(1, 1, device=device)

                                    if mode in ("our", "our_v2", "our_v3", "ablation_no_dib", "ablation_no_it"):
                                        s_t_val, _, _ = agent.encoder(val_state, deterministic=True)
                                        if getattr(agent, 'ablation_mode', None) == 'no_it':
                                            internal_action_it = torch.zeros(1, 1, device=device)
                                        else:
                                            internal_action_it, _ = agent.actor_it(s_t_val, deterministic=True, with_logprob=False)
                                        s_worker_val = torch.cat([s_t_val, internal_action_it], dim=-1)
                                        action_mm_raw, _ = agent.actor_mm(s_worker_val, deterministic=True, with_logprob=False)

                                        p1 = val_env.lob_data[val_env.current_tick, 0]
                                        p2 = val_env.lob_data[val_env.current_tick, 2]
                                        mid_val = (p1 + p2) / 2.0
                                        action_as = as_prior_generator.get_action(mid_val, val_env.inventory, current_tick_size, max_offset=adaptive_max_offset)
                                        
                                        if LOADER_TYPE in ['lob_bench', 'trades_lob', 'a_share', 'fi2010']:
                                            action_mm_val = torch.clamp(torch.abs(action_mm_raw[..., :2]) * config['action_scale'] * 2.0, 0.01, 3.0)
                                        else:
                                            action_mm_val = blend_action(action_mm_raw, action_as, config['action_scale'], max_gate=current_max_gate, min_val=0.0)

                                    elif mode in ["sac", "deeplob", "timesfm", "moment", "timer_xl"]:
                                        s_t_val = agent.encoder(val_state)
                                        action_mm_val, _ = agent.actor_mm(s_t_val, deterministic=True, with_logprob=False)

                                    elif mode == "ppo":
                                        s_t_val = agent.encoder(val_state)
                                        action_mm_val, _ = agent.actor_mm(s_t_val, deterministic=True, with_logprob=False)

                                    elif mode == "hrl":
                                        s_t_val = agent.encoder(val_state)
                                        goal_val, _ = agent.manager(s_t_val)
                                        s_worker_val = torch.cat([s_t_val, goal_val], dim=-1)
                                        action_mm_val, _ = agent.actor_mm(s_worker_val, deterministic=True, with_logprob=False)

                                    elif mode == "mappo":
                                        s_t_val = agent.encoder(val_state)
                                        action_mm_val, _ = agent.actor_mm(s_t_val, deterministic=True, with_logprob=False)

                                next_val_state, val_reward, val_done = val_env.step(action_mm_val, action_it_val, adv_decay=0.0)
                                val_inventories.append(val_env.inventory)

                                p1 = val_env.lob_data[val_env.current_tick - 1, 0]
                                p2 = val_env.lob_data[val_env.current_tick - 1, 2]
                                mid_val = (max(p1, p2) + min(p1, p2)) / 2.0
                                p_val = val_env.cash + val_env.inventory * mid_val
                                portfolio_history.append(p_val)

                                if step_cnt % 5000 == 0:
                                    print(f"[VAL TICK PROGRESS] {step_cnt}/{total_val_ticks} Inventory:{val_env.inventory:.2f} Wealth:{p_val:.2f}", flush=True)

                                val_state = next_val_state
                            val_sharpe_real = calculate_real_daily_sharpe(
                                portfolio_history=portfolio_history,
                                dataset_type=DATASET_TYPE,
                                bar_size=2000,
                                trade_count=val_env.trade_count,
                                initial_capital=adaptive_capital
                            )

                            full_nav = np.array(portfolio_history)
                            cum_max = np.maximum.accumulate(full_nav)
                            denominator = np.maximum(cum_max, adaptive_capital)
                            drawdowns = (cum_max - full_nav) / (denominator + 1e-12)
                            val_max_dd = np.clip(np.max(drawdowns), 0.0, 1.0)

                            val_inventories_var = float(np.var(val_inventories))
                            val_utility = val_sharpe_real - 0.05 * val_inventories_var

                            print(
                                f"==> Epoch {epoch:2d} (SEED {SEED_VAL}) 结束 | "
                                f"验证效用 (Utility): {val_utility:.4f} | "
                                f"夏普 (Sharpe): {val_sharpe_real:.4f} | "
                                f"方差 (Variance): {val_inventories_var:.4f} | "
                                f"最大回撤: {val_max_dd * 100:.2f}%",
                                flush=True
                            )

                            if val_utility > best_val_utility:
                                best_val_utility = val_utility
                                best_portfolio_history = list(portfolio_history)
                                best_val_inventories = list(val_inventories)
                                best_model_state = {
                                    "encoder": agent.encoder.state_dict(),
                                    "actor_mm": agent.actor_mm.state_dict(),
                                    "epoch": epoch,
                                    "val_utility": val_utility,          
                                    "val_sharpe": val_sharpe_real,
                                    "val_inventory_variance": val_inventories_var,
                                    "val_max_dd": val_max_dd,
                                    "trades": val_env.trade_count
                                }
                                if hasattr(agent, "manager"):
                                    best_model_state["manager"] = agent.manager.state_dict()

                                torch.save(best_model_state, final_model_path)

                                df_traj = pd.DataFrame({
                                    "step": range(len(portfolio_history)),
                                    "portfolio_value": portfolio_history,
                                    "inventory": val_inventories
                                })
                                df_traj.to_csv(final_traj_path, index=False)

                        if os.path.exists(final_model_path):
                            checkpoint = torch.load(final_model_path, map_location="cpu", weights_only=False)
                            best_ut = checkpoint.get("val_utility", -999.0)
                            best_sh = checkpoint.get("val_sharpe", -999.0)
                            best_var = checkpoint.get("val_inventory_variance", 999.0)
                            best_dd = checkpoint.get("val_max_dd", 1.0)
                            best_trades = checkpoint.get("trades", 0)

                            seed_elapsed = time.time() - seed_start_time
                            save_result_to_csv(
                                progress_log_file, DATASET_TYPE, mode, SEED_VAL,
                                best_ut, best_sh, best_var, best_dd, best_trades,
                                start_time_str, seed_elapsed, is_our, output_dir
                            )
                            if len(best_portfolio_history) > 0:
                                plot_and_save_trajectory(best_portfolio_history, best_val_inventories, mode, DATASET_TYPE, SEED_VAL, output_dir, run_timestamp)
                                
                            dataset_summary_stats[DATASET_TYPE][mode].append(best_ut)
                            test_results[f"{DATASET_TYPE}_{mode}_seed{SEED_VAL}"] = f"✅ 学术跑数成功 (最佳 Utility: {best_ut:.4f})"

                        if not is_heuristic and epoch > 5:
                            decay_factor = 0.95 ** (epoch - 5)
                            if mode == "our":
                                for g in agent.critic_opt.param_groups:
                                    g['lr'] = LR * decay_factor
                                for g in agent.actor_mm_opt.param_groups:
                                    g['lr'] = LR * decay_factor
                                for g in agent.actor_it_opt.param_groups:
                                    g['lr'] = (LR * 0.33) * decay_factor
                            elif hasattr(agent, "critic_opt") and hasattr(agent, "actor_mm_opt"):
                                for g in agent.critic_opt.param_groups:
                                    g['lr'] = LR * decay_factor
                                for g in agent.actor_mm_opt.param_groups:
                                    g['lr'] = LR * decay_factor

                    except Exception as seed_err:
                        print(f"\n❌ [种子级错误警报] 模式 {mode.upper()} 种子 {SEED_VAL} 异常！", flush=True)
                        traceback.print_exc()
                        test_results[f"{DATASET_TYPE}_{mode}_seed{SEED_VAL}"] = f"❌ 运行失败 (SEED ERROR: {str(seed_err)})"

                    finally:
                        if 'agent' in locals(): del agent
                        if 'buffer' in locals(): del buffer
                        if 'env' in locals(): del env
                        if 'val_env' in locals(): del val_env
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

        except Exception as e:
            test_results[DATASET_TYPE] = f"❌ 数据集发生崩溃 (FAILED: {str(e)})"
            traceback.print_exc()

        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("-" * 60, flush=True)

    # ======================================================================
    # 8. 学术级数据汇总 (均值-方差效用方向)
    # ======================================================================
    print("\n" + "=" * 95, flush=True)
    print(f"📊 【DIB-MASAC 多模式/多种子实验 - 学术结算大盘（均值-方差效用 κ=0.05）】", flush=True)
    print("=" * 95, flush=True)
    
    header = f"{'Target Market':<24}"
    for m in modes_to_test:
        header += f" | {m.upper():^14}"
    print(header, flush=True)
    print("-" * 95, flush=True)
    
    for market in datasets_to_test:
        row = f"{market:<24}"
        for m in modes_to_test:
            sharpe_list = dataset_summary_stats[market][m]
            if len(sharpe_list) > 0:
                mean_val = np.mean(sharpe_list)
                std_val = np.std(sharpe_list, ddof=1) if len(sharpe_list) > 1 else 0.0
                row += f" | \033[1;32m{mean_val:6.4f}±{std_val:5.4f}\033[0m"
            else:
                row += f" | {'N/A':^14}"
        print(row, flush=True)
    print("=" * 95 + "\n", flush=True)


if __name__ == "__main__":
    main()