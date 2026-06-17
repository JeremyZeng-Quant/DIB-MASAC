import os
import glob
import random
import argparse
import traceback
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

# 忽略不必要的冗余警告，保持终端学术输出的干净整洁
warnings.filterwarnings("ignore")

# ========== 全局超参数（直接作为论文参数表复制） ==========
SEED = 42
TAU = 0.995            # 目标网络软更新系数 (软更新超参)
LR = 3e-4              # 统一优化器学习率
GAMMA = 0.99           # 强化学习折扣因子
CLIP_EPS = 0.2         # PPO 裁剪概率范围
ALPHA_0 = 0.2          # SAC 熵正则化初始温度系数
KL_COEF = 1e-4         # DIB 表征瓶颈 KL 散度约束系数
BETA = 0.1             # DIB 辅助任务奖励预测损失权重
G_MAX = 1.0            # 梯度裁剪硬限
MAX_INVENTORY = 50     # 策略最大允许绝对持仓限制
SEQ_LEN = 5            # 做市决策 LOB 特征回溯窗口步数

# 🎯 提速版核心参数：充分榨干 L40S 显卡的强大并行计算能力，且将反向传播频率大幅压缩 [1]
BATCH_SIZE = 512       # 神经网络训练批次大小 (已升级，充分压榨 VRAM)
UPDATE_EVERY = 150     # 经验回放训练更新频率 (已升级，梯度更新频次减少 15 倍)
# ====================================================

# ====================== 固定全局随机种子（确保学术可复现性） ======================
random.seed(SEED)
np.random.seed(SEED)  # 已移除原无效的 np.seed = SEED 属性错误
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.distributions.Distribution.set_default_validate_args(False)
# ======================================================================

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# 1. 运行模式选择：False = 本地极速联调测试；True = 样本外评估/正式生产跑数（论文正式结果开启 True）
EVALUATE_MODE = True

# 2. 强行限制载入总 Tick 数（大盘限制在 50 万连续 Tick，这在学术论文里是常规且合理的）
MAX_TICKS = 500000 if EVALUATE_MODE else 50000

# 3. 每个数据集跑的默认 Epoch 数量（后续在 main() 中会自动根据数据集大小被自适应局部变量安全覆盖）
MAX_EPOCHS = 30 if EVALUATE_MODE else 5

# 🎯 各市场细分数据集的真实最小报价单位 (Tick Size) 映射字典
TICK_SIZES = {
    "fi2010": 0.01,
    "lob_bench_goog_real": 0.01,
    "lob_bench_goog_synth": 0.01,
    "lob_bench_intc_real": 0.01,
    "lob_bench_intc_synth": 0.01,
    "binance_low_vol": 0.1,      # 高频数字货币做市真实报价单位
    "binance_high_vol": 0.1,
    "trades_lob_tsla": 0.01,
    "trades_lob_intc": 0.01,
    "a_share_sz000001": 0.01,
    "a_share_sz000651": 0.01,
    "a_share_sz002415": 0.01,
    "a_share_sz300147": 0.01,
}

device = torch.device("cpu")  # 默认设备，将在 main 中根据参数动态重定向


# ====================== GPU 驻留 Replay Buffer（消除 PCIe 拷贝瓶颈） ======================
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

        # 已修复 ReplayBuffer 形状与维度兼容性 Bug（经典模型与多智能体完全对齐）
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
# 1. 表征编码器（Encoders）集合
# ======================================================================

# --- Deep Information Bottleneck (DIB) Encoder (Ours) ---
class DIBLOBEncoder(nn.Module):
    def __init__(self, channels=1, latent_dim=32):
        super(DIBLOBEncoder, self).__init__()
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8)
        )
        self.temporal_gru = nn.GRU(input_size=64 * 8, hidden_size=128, batch_first=True)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, L_t):
        batch_size, seq_len, features = L_t.shape
        x = L_t.view(batch_size * seq_len, 1, features)
        x = self.spatial_conv(x)
        x = x.view(batch_size, seq_len, -1)
        _, h = self.temporal_gru(x)
        h = h[0]
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)

        mu = torch.clamp(mu, -10.0, 10.0)
        log_var = torch.clamp(log_var, -8.0, 4.0)

        s_t = self.reparameterize(mu, log_var)
        return s_t, mu, log_var


# --- TimesFM Backbone Encoder ---
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


# --- MOMENT Backbone Encoder ---
class MomentEncoder(nn.Module):
    def __init__(self, d_model=128, latent_dim=32):
        super(MomentEncoder, self).__init__()
        self.projection = nn.Linear(41, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=256, batch_first=True, norm_first=True),
            num_layers=2
        )
        self.fc_out = nn.Linear(d_model, latent_dim)

    def forward(self, L_t):
        batch_size, seq_len, features = L_t.shape
        x_proj = self.projection(L_t)
        out = self.transformer(x_proj)
        s_t = self.fc_out(out.mean(dim=1))
        return s_t


# --- Timer-XL Backbone Encoder ---
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


# --- DeepLOB Encoder (LOB-Specific CNN-LSTM) ---
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
            log_prob = torch.clamp(log_prob, -50.0, 10.0) # 已放宽边界至 -50.0 防止高频梯度卡死
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
# 3. 策略与对比基线算法类
# ======================================================================

# --- Avellaneda-Stoikov 经典报价模型 (自适应校准版) ---
class AvellanedaStoikovAgent:
    def __init__(self, gamma=0.1, kappa=1.5, window_size=100):
        self.gamma = gamma
        self.kappa = kappa
        self.window_size = window_size
        self.mid_history = []

    def get_action(self, mid_price, inventory, tick_size):
        self.mid_history.append(mid_price)
        if len(self.mid_history) > self.window_size:
            self.mid_history.pop(0)

        if len(self.mid_history) < 2:
            sigma = 0.01
        else:
            returns = np.diff(self.mid_history)
            sigma = np.std(returns) if np.std(returns) > 1e-6 else 0.01

        adaptive_kappa = self.kappa / tick_size

        r = mid_price - inventory * self.gamma * (sigma ** 2)
        spread = self.gamma * (sigma ** 2) + (2 / self.gamma) * np.log(1 + self.gamma / adaptive_kappa)
        
        my_ask = r + spread / 2.0
        my_bid = r - spread / 2.0

        act_mm_0 = max(0.5, (my_ask - mid_price) / tick_size)
        act_mm_1 = max(0.5, (mid_price - my_bid) / tick_size)
        return torch.tensor([[act_mm_0, act_mm_1]], dtype=torch.float32).to(device)


# --- Ho-Stoll 微观经纪商模型 (安全保护版) ---
class HoStollAgent:
    def __init__(self, gamma=0.1, window_size=100):
        self.gamma = gamma
        self.window_size = window_size
        self.mid_history = []

    def get_action(self, mid_price, inventory, tick_size):
        self.mid_history.append(mid_price)
        if len(self.mid_history) > self.window_size:
            self.mid_history.pop(0)

        if len(self.mid_history) < 2:
            sigma = 0.01
        else:
            returns = np.diff(self.mid_history)
            sigma = np.std(returns) if np.std(returns) > 1e-6 else 0.01

        r_ask = mid_price + (1 + 2 * inventory) * 0.5 * self.gamma * (sigma ** 2)
        r_bid = mid_price - (1 - 2 * inventory) * 0.5 * self.gamma * (sigma ** 2)

        act_mm_0 = max(1.0, (r_ask - mid_price) / tick_size)
        act_mm_1 = max(1.0, (mid_price - r_bid) / tick_size)
        return torch.tensor([[act_mm_0, act_mm_1]], dtype=torch.float32).to(device)


# --- PPO (Single) 算法实现 ---
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


# --- Hierarchical-RL (HRL) 算法实现 ---
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
        
        # 补全 HRL 原生演员更新梯度损失逻辑，实现完整的双层优化
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

        # 1. Critic 更新
        q1 = self.critic_1(s_t, action_mm_hist, action_it_hist)
        q2 = self.critic_2(s_t, action_mm_hist, action_it_hist)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        
        # 2. Worker Actor (actor_mm) 更新 
        act_mm, log_prob_mm = self.actor_mm(s_worker)
        q1_pi_worker = self.critic_1(s_t, act_mm, torch.zeros_like(action_it_hist))
        q2_pi_worker = self.critic_2(s_t, act_mm, torch.zeros_like(action_it_hist))
        q_pi_worker = torch.min(q1_pi_worker, q2_pi_worker)
        actor_mm_loss = (ALPHA_0 * log_prob_mm - q_pi_worker).mean()

        # 3. Manager Actor 更新
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


# --- Multi-Agent MAPPO 经典博弈基线 ---
class StandardMAPPO:
    def __init__(self, state_dim=32, action_dim_mm=2, action_dim_it=1, lr=LR, clip_eps=CLIP_EPS, gamma=GAMMA, G_max=G_MAX):
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


# --- DIB-MASAC (主模型) ---
class StabilizedAdversarialMASAC:
    def __init__(self, state_dim=32, action_dim_mm=2, action_dim_it=1, lr=LR, beta=BETA, alpha_0=ALPHA_0, gamma=GAMMA,
                 G_max=G_MAX, kl_coef=KL_COEF):
        self.gamma = gamma
        self.beta = beta
        self.alpha_0 = alpha_0
        self.G_max = G_max
        self.kl_coef = kl_coef 

        self.encoder = DIBLOBEncoder(channels=1, latent_dim=state_dim).to(device)
        self.actor_mm = GaussianActor(state_dim, action_dim_mm).to(device)
        self.actor_it = GaussianActor(state_dim, action_dim_it).to(device)
        self.critic_1 = CentralizedCritic(state_dim, action_dim_mm, action_dim_it).to(device)
        self.critic_2 = CentralizedCritic(state_dim, action_dim_mm, action_dim_it).to(device)

        self.reward_decoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(device)

        self.encoder_target = DIBLOBEncoder(channels=1, latent_dim=state_dim).to(device)
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
            
            next_act_mm, log_prob_next_mm = self.actor_mm(s_next)
            next_act_it, log_prob_next_it = self.actor_it(s_next)

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
            alpha = self.alpha_0 * torch.exp(-1.0 * exploitability)

            y = r_t + (1 - done) * self.gamma * (target_Q - alpha * (log_prob_next_mm + log_prob_next_it))
            y = torch.clamp(y, -50.0, 50.0)

        q1 = self.critic_1(s_t, action_mm_hist, action_it_hist)
        q2 = self.critic_2(s_t, action_mm_hist, action_it_hist)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1).mean()
        kl_div = torch.clamp(kl_div, 1e-6, 20.0) # 已增加下界保护防止散度爆炸
        pred_reward = self.reward_decoder(s_t)
        prediction_loss = F.mse_loss(pred_reward, r_t)
        
        dib_loss = self.kl_coef * kl_div + self.beta * prediction_loss
        total_critic_loss = critic_loss + dib_loss

        self.opt.zero_grad() if hasattr(self, 'opt') else self.critic_opt.zero_grad()
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

        act_mm, log_prob_mm = self.actor_mm(s_t_detached)
        act_it, log_prob_it = self.actor_it(s_t_detached)

        q1_pi_mm = self.critic_1(s_t_detached, act_mm, act_it.detach())
        q2_pi_mm = self.critic_2(s_t_detached, act_mm, act_it.detach())
        q_pi_mm = torch.min(q1_pi_mm, q2_pi_mm)

        actor_mm_loss = (self.alpha_0 * log_prob_mm - q_pi_mm).mean()
        self.actor_mm_opt.zero_grad()
        actor_mm_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_mm.parameters(), self.G_max)
        self.actor_mm_opt.step()

        q1_pi_it = self.critic_1(s_t_detached, act_mm.detach(), act_it)
        q2_pi_it = self.critic_2(s_t_detached, act_mm.detach(), act_it)
        q_pi_it = torch.min(q1_pi_it, q2_pi_it)

        actor_it_loss = (self.alpha_0 * log_prob_it + q_pi_it).mean()
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


# --- 通用单智能体 SAC 架构 (集成大模型与专用表征 Backbone) ---
class GeneralizedSACBaseline:
    def __init__(self, encoder_type="sac", state_dim=32, action_dim_mm=2, lr=LR, alpha_0=ALPHA_0, gamma=GAMMA, G_max=G_MAX):
        self.gamma = gamma
        self.alpha_0 = alpha_0 # 已修正：删除了错误覆盖 alpha_0 的冗余行
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
        self.current_tick = self.seq_len
        self.cash = self.initial_capital
        self.inventory = 0.0
        self.trade_count = 0
        return self._get_state()

    def _get_state(self):
        start_idx = self.current_tick - self.seq_len
        end_idx = self.current_tick

        state = self.lob_data[start_idx:end_idx].copy().astype(np.float32)

        mid_prices = (state[:, 0] + state[:, 2]) / 2.0
        jump_idx = -1
        for i in range(1, len(mid_prices)):
            prev_mid = mid_prices[i-1]
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

        for j in range(10):
            state_normalized[:, j * 4] = (state_normalized[:, j * 4] - mid) / self.tick_size
            state_normalized[:, j * 4 + 2] = (state_normalized[:, j * 4 + 2] - mid) / self.tick_size

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

    def step(self, action_mm, action_it):
        p1 = self.lob_data[self.current_tick, 0]
        p2 = self.lob_data[self.current_tick, 2]
        ask_1 = max(p1, p2)
        bid_1 = min(p1, p2)
        mid_price = (ask_1 + bid_1) / 2.0

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

        my_ask_price = ask_1 + round(offset_ask_ticks) * self.tick_size
        my_bid_price = bid_1 - round(offset_bid_ticks) * self.tick_size

        my_ask_price = max(my_ask_price, bid_1 + self.tick_size)
        my_bid_price = min(my_bid_price, ask_1 - self.tick_size)

        my_ask_price = max(1e-5, my_ask_price)
        my_bid_price = max(1e-5, my_bid_price)

        next_tick = self.current_tick + 1
        if next_tick >= len(self.lob_data):
            next_tick = self.current_tick

        next_p1 = self.lob_data[next_tick, 0]
        next_p2 = self.lob_data[next_tick, 2]
        next_ask = max(next_p1, next_p2)
        next_bid = min(next_p1, next_p2)

        if next_ask > my_ask_price:
            fill_ask = True
        elif next_ask == my_ask_price:
            fill_ask = (np.random.rand() < 0.15)
        else:
            fill_ask = False

        if next_bid < my_bid_price:
            fill_bid = True
        elif next_bid == my_bid_price:
            fill_bid = (np.random.rand() < 0.15)
        else:
            fill_bid = False

        fill_ask_adv = False
        if a_it_val > 0.1:
            dist_ask_ticks = max(0.0, (my_ask_price - ask_1) / self.tick_size)
            prob_adv = a_it_val * np.exp(-0.5 * dist_ask_ticks)
            fill_ask_adv = (np.random.rand() < prob_adv)

        fill_bid_adv = False
        if a_it_val < -0.1:
            dist_bid_ticks = max(0.0, (bid_1 - my_bid_price) / self.tick_size)
            prob_adv = abs(a_it_val) * np.exp(-0.5 * dist_bid_ticks)
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
        r_penalty = -5.0 * (norm_inv ** 2)

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

        if self.normalize_reward:
            max_exposure = max(self.max_inventory * mid_price, 1e-6)
            r_wealth = (raw_r_wealth / max_exposure) * 10000.0
        else:
            r_wealth = raw_r_wealth

        r_wealth = np.clip(r_wealth, -10.0, 10.0)

        if is_bankrupt:
            liquidation_fee -= 10.0

        total_raw_reward = r_wealth + r_penalty + liquidation_fee
        # 统一全局学术标准截断奖励范围 [-3.0, 3.0]，提升高波动标的下智能体训练的策略稳定性
        reward_scaled = np.clip(total_raw_reward, -3.0, 3.0)
        reward = torch.tensor([[reward_scaled]], dtype=torch.float32).to(device)

        next_state = self._get_state()

        return next_state, reward, done


# ====================== 5. 科学自适应夏普计算函数（日均夏普校准版） ======================
def calculate_real_daily_sharpe(portfolio_history, dataset_type, bar_size=2000,
                                 trade_count=0, initial_capital=10000.0):
    """
    计算高频做市策略的日均夏普比率 (Daily Sharpe Ratio)。
    在超短周期回测中，日均夏普比率比直接外推年化夏普更具统计学严谨性，能有效避免高频自相关导致的指标膨胀。
    """
    if trade_count < 10:
        return 0.0

    nav_array = np.array(portfolio_history)
    total_ticks = len(nav_array)
    
    # 🎯 动态自适应重采样区间
    if total_ticks >= 200:
        bar_size = max(5, total_ticks // 100)
    else:
        bar_size = 2
        
    if total_ticks < bar_size * 2:
        return 0.0

    # 动态重采样
    sampled_nav = nav_array[::bar_size]
    delta_nav_raw = sampled_nav[1:] - sampled_nav[:-1]
    
    if len(delta_nav_raw) < 5:
        return 0.0

    mean_pnl = np.mean(delta_nav_raw)
    variance = np.var(delta_nav_raw, ddof=1)
    
    if variance <= 1e-12:
        return 0.0
        
    # 🎯 恢复纯粹标准数理夏普公式，仅使用 1e-8 作为微小的防除零极小项保护
    std_pnl = np.sqrt(variance) + 1e-8

    # 🎯 将高频 bar 的收益率标准校准到【日频】(Daily Scaling Factor)
    if "binance" in dataset_type.lower():
        ticks_per_day = 20000.0   # Binance 做市环境日均 Tick 总数
    else:
        ticks_per_day = 4800.0    # A股等 LOB 市场日均 Tick 总数

    bars_per_day = ticks_per_day / bar_size
    scale_factor_daily = np.sqrt(bars_per_day)

    # 🎯 计算无人工扭曲的日均夏普比率 (Daily Sharpe)
    raw_daily_sharpe = (mean_pnl / std_pnl) * scale_factor_daily
    
    # 🎯 温和的小样本修正系数
    sample_size = len(delta_nav_raw)
    small_sample_correction = np.sqrt(sample_size / (sample_size + 15.0))
    
    final_daily_sharpe = raw_daily_sharpe * small_sample_correction

    # 限制合理学术上限 [-100.0, 100.0]，防止局部异常导致的指标溢出，拓宽截断支持多算法精细比对
    final_daily_sharpe = float(np.clip(final_daily_sharpe, -100.0, 100.0))
    
    return final_daily_sharpe


# ======================================================================
# 6. 通用 LOB 数据加载与“数据防崩卫士”对齐接口
# ======================================================================
def load_lob_data(filepath, dataset_type="fi2010", max_ticks=None, is_real_target=True):
    if filepath is not None and os.path.exists(filepath):
        print(f"\n正在加载【{dataset_type.upper()}】高频 LOB 数据: {filepath} ...")

        if dataset_type == "a_share":
            cols = []
            for i in range(1, 11):
                cols.extend([f'AskPrice{i}', f'AskVolume{i}', f'BidPrice{i}', f'BidVolume{i}'])

            if max_ticks is not None:
                df = pd.read_csv(filepath, usecols=cols, nrows=max_ticks + 100)
            else:
                df = pd.read_csv(filepath, usecols=cols)

            raw_data_t = df[cols].values
            print("   👉 成功完成 A股 专属数据列格式对齐。")

        elif dataset_type == "lob_bench" and os.path.isdir(filepath):
            all_csvs = glob.glob(os.path.join(filepath, "**", "*.csv"), recursive=True)
            orderbook_files = [f for f in all_csvs if "orderbook" in os.path.basename(f).lower()]

            if len(orderbook_files) == 0:
                raise FileNotFoundError(
                    f"❌ [数据验证失败] 在目录 {filepath} 下无法检索到任何做市订单簿 (orderbook) 数据文件。"
                )

            matched_files = sorted(orderbook_files)
            print(f"   👉 成功匹配并定位到 {len(matched_files)} 个 LOB-Bench 订单簿分卷，合并加载中...")
            dfs = []
            for f in matched_files[:15]:
                dfs.append(pd.read_csv(f, header=None))
            df = pd.concat(dfs, ignore_index=True)

            raw_values = df.values[:, :40]
            reordered_cols = []
            for i in range(10):
                reordered_cols.extend([i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3])
            raw_data_t = raw_values[:, reordered_cols]
            print("   👉 成功完成 LOB-Bench 数据列格式对齐。")

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
            print("   👉 成功从本地 PyTorch .pt 预处理张量文件加载并转换。")

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
            print("   👉 成功完成 Binance 数据列格式对齐（按 LOBSTER 标准）。")
        else:
            raise ValueError(f"未知的数据集类型: {dataset_type}")

        raw_data_t = np.asarray(raw_data_t, dtype=np.float32)

        is_normalized = (raw_data_t[:, [0, 2]] < 0.0).any() or (np.abs(raw_data_t[:, [0, 2]].mean()) < 5.0)

        # 增强 NaN 防护逻辑，同步对 LOB 挂单量进行空值填充并裁剪负值，提升数据平稳性
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
            print("   ⚠️ [数据卫士] 检测到输入数据已完成 Z-score/Min-Max 归一化。")
            print("   🔄 启动【自适应绝对特征重构】，将其无损回归至真实交易引擎的价格与挂单量物理空间...")
            
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
            
            print("   👉 [绝对特征重构成功] 所有价格列已无损映射到 $100 基准空间，挂单量已映射回物理空间。")
        else:
            for col_idx in [0, 2]:
                bad_mask = (raw_data_t[:, col_idx] <= 1e-6) | np.isnan(raw_data_t[:, col_idx]) | np.isinf(raw_data_t[:, col_idx])
                if bad_mask.any():
                    print(f"   ⚠️ [数据卫士] 检测到第 {col_idx} 列存在 {bad_mask.sum()} 个异常价格 Tick，执行前向填充...")
                    temp_series = pd.Series(raw_data_t[:, col_idx])
                    temp_series[bad_mask] = np.nan
                    temp_series = temp_series.ffill().bfill().fillna(1.0)
                    raw_data_t[:, col_idx] = temp_series.values

            if raw_data_t[:, 0].mean() > 10000.0 and dataset_type in ["lob_bench", "trades_lob"]:
                print("   ⚠️ [数据卫士] 检测到特定格式价格时序量级过大（均价 > 10,000），自动还原 10,000 倍标的价格。")
                for col_idx in range(40):
                    if col_idx % 2 == 0:
                        raw_data_t[:, col_idx] /= 10000.0
            else:
                if raw_data_t[:, 0].mean() > 10000.0:
                    print(f"   ℹ️ [数据卫士] 检测到高价值物理标的价格 ({raw_data_t[:, 0].mean():.2f})，确认为真实交易空间，跳过价格还原。")

            for col_idx in range(40):
                if col_idx % 2 == 1:
                    raw_data_t[:, col_idx] = np.clip(raw_data_t[:, col_idx], 0.0, None)

        if max_ticks is not None and len(raw_data_t) > max_ticks:
            raw_data_t = raw_data_t[:max_ticks]
            print(f"   🚀 已启用极速切片！只截取前 {max_ticks} 个交易 Tick。")
        else:
            print(f"   🚀 已启用全量数据集！载入 {len(raw_data_t)} 个交易 Ticks 运行。")

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
        choices=["all", "fi2010",
                 "lob_bench_goog", "lob_bench_intc",
                 "lob_bench_goog_real", "lob_bench_goog_synth", "lob_bench_intc_real", "lob_bench_intc_synth",
                 "binance_low_vol", "binance_high_vol",
                 "trades_lob_tsla", "trades_lob_intc",
                 "a_share_sz000001", "a_share_sz000651", "a_share_sz002415", "a_share_sz300147"],
        help="指定运行的目标市场。默认 'a_share_sz000001'。"
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
        choices=["our", "as", "ho_stoll", "ppo", "sac", "hrl", "mappo", "deeplob", "timesfm", "moment", "timer_xl"],
        help="运行模式"
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        gpu_id = min(max(0, args.gpu), torch.cuda.device_count() - 1)
        device = torch.device(f"cuda:{gpu_id}")
        # 在主程序 GPU 卡号重定向后，延迟执行网络模块向指定 CUDA 设备的迁移绑定，确保完全避开CPU/GPU跨设备通信开销
        torch.cuda.set_device(device)
    print(f"当前运行激活设备 (Device): {device} | 运行模式: 【{args.mode.upper()}】")

    all_datasets = [
        "fi2010",
        "lob_bench_goog_real", "lob_bench_goog_synth",
        "lob_bench_intc_real", "lob_bench_intc_synth",
        "binance_low_vol", "binance_high_vol",
        "trades_lob_tsla", "trades_lob_intc",
        "a_share_sz000001", "a_share_sz000651", "a_share_sz002415", "a_share_sz300147"
    ]

    target_dataset = args.dataset

    if args.dataset == "all":
        datasets_to_test = all_datasets
    else:
        datasets_to_test = [target_dataset]
        print(f"🚀 专项训练和评估市场：【{target_dataset.upper()}】")

    test_results = {}

    print("\n" + "=" * 65)
    print(f"🚀 【数据对齐重隔版：DIB-MASAC 实验系统】启动 | 当前模式: {args.mode.upper()}")
    print("=" * 65)

    for DATASET_TYPE in datasets_to_test:
        print("\n" + "#" * 60)
        print(f"🎬 [市场实验阶段] 开始跑数数据集: {DATASET_TYPE.upper()}")
        print("#" * 60)

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
                ACTION_SCALE = 10.0
                LOADER_TYPE = "binance"

            elif DATASET_TYPE == "binance_high_vol":
                matched_file = find_file_path("binance_high_vol.pt") or find_file_path("binance_high_vol_toy.pt")
                TRAIN_FILE = matched_file if matched_file else "binance_high_vol_toy.pt"
                VAL_FILE = TRAIN_FILE
                ACTION_SCALE = 10.0
                LOADER_TYPE = "binance"

            elif "lob_bench" in DATASET_TYPE:
                asset_dir = "GOOG" if "goog" in DATASET_TYPE else "INTC"
                base_dir = find_file_path("lob_bench_data")
                if base_dir and os.path.exists(os.path.join(base_dir, asset_dir)):
                    TRAIN_FILE = os.path.join(base_dir, asset_dir)
                else:
                    TRAIN_FILE = (
                            find_file_path(f"*lob_bench_{asset_dir.lower()}*.pt") or
                            find_file_path(f"*{asset_dir.lower()}*_real.pt") or
                            find_file_path(f"*{asset_dir.lower()}*_synth.pt") or
                            find_file_path(f"*{asset_dir.lower()}*.pt") or
                            f"{DATASET_TYPE}_toy.pt"
                    )
                VAL_FILE = TRAIN_FILE
                ACTION_SCALE = 1.0
                LOADER_TYPE = "lob_bench"
            else:
                raise ValueError(f"未知数据集类型: {DATASET_TYPE}")

            config = {
                "max_inventory": 50.0,
                "action_scale": ACTION_SCALE,
            }

            # ====================== 1. 实验代理选择初始化分流 ======================
            is_heuristic = args.mode in ["as", "ho_stoll"]

            if args.mode == "our":
                agent = StabilizedAdversarialMASAC(state_dim=32, action_dim_mm=2, action_dim_it=1)
            elif args.mode == "as":
                agent = AvellanedaStoikovAgent(gamma=0.1, kappa=1.5)
            elif args.mode == "ho_stoll":
                agent = HoStollAgent(gamma=0.1)
            elif args.mode == "ppo":
                agent = SinglePPOAgent(state_dim=32, action_dim_mm=2)
            elif args.mode == "hrl":
                agent = HierarchicalRLAgent(state_dim=32, action_dim_mm=2)
            elif args.mode == "mappo":
                agent = StandardMAPPO(state_dim=32, action_dim_mm=2, action_dim_it=1)
            elif args.mode in ["sac", "deeplob", "timesfm", "moment", "timer_xl"]:
                agent = GeneralizedSACBaseline(encoder_type=args.mode, state_dim=32, action_dim_mm=2)
            else:
                raise ValueError(f"未知的运行模式: {args.mode}")

            # 延迟迁移绑定：在 GPU 设备号重定向 main 后执行统一设备迁移
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

            # 尝试加载历史最佳权重 (🎯 完美适应所有神经网络模式与 HRL 专用权重载入)
            if not is_heuristic and agent is not None:
                model_path = find_file_path(f"best_hfmm_model_{args.mode}_{DATASET_TYPE}.pth")
                if model_path and os.path.exists(model_path):
                    print(f"   🔄 [初始化恢复] 检测到本地已存在最佳权重备份: {model_path}")
                    try:
                        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                        agent.encoder.load_state_dict(checkpoint["encoder"])
                        agent.actor_mm.load_state_dict(checkpoint["actor_mm"])
                        # 🎯 修复处 2：如果是分层强化学习，自适应加载 manager 高层网络的权重
                        if "manager" in checkpoint and hasattr(agent, "manager"):
                            agent.manager.load_state_dict(checkpoint["manager"])
                        print(f"   👉 [恢复成功] 历史最佳日均夏普为: {checkpoint['val_sharpe']:.4f}")
                    except Exception:
                        print("   ⚠️ [恢复失败] 无法加载，随机初始化启动。")
                else:
                    print("   👉 [随机启动] 无最佳模型备份，采用全新的随机初始化。")

            is_real = "real" in DATASET_TYPE
            train_lob = load_lob_data(filepath=TRAIN_FILE, dataset_type=LOADER_TYPE, max_ticks=MAX_TICKS,
                                      is_real_target=is_real)

            if TRAIN_FILE == VAL_FILE:
                split_idx = int(len(train_lob) * 0.8)
                val_lob = train_lob[split_idx:]
                train_lob = train_lob[:split_idx]
                print("   ⚠️ 已自动执行 Train/Val 单向切分 (80% 训练，20% 样本外验证)。")
            else:
                val_lob = load_lob_data(filepath=VAL_FILE, dataset_type=LOADER_TYPE, max_ticks=MAX_TICKS,
                                        is_real_target=is_real)

            print(f"数据切分完成 -> 训练集 Ticks: {len(train_lob)} | 样本外测试集 (Val) Ticks: {len(val_lob)}")

            # =================================================================================
            # 🎯【自适应 Epoch 局部决策点】：使用当前加载的 train_lob 动态决定 Epoch 长度 [3]
            # =================================================================================
            epochs_to_run = MAX_EPOCHS
            if EVALUATE_MODE:
                if len(train_lob) > 500000:
                    # 大数据集（如 A 股、币安）只跑 5 个 Epoch，避免时间过载 [1]
                    epochs_to_run = 5 
                elif len(train_lob) > 200000:
                    # 中等数据集（如 fi2010）跑 12 个 Epoch
                    epochs_to_run = 12
                else:
                    # 小数据集（如谷歌、英特尔）跑 30 个 Epoch
                    epochs_to_run = 30
            else:
                epochs_to_run = 3
            # =================================================================================

            batch_size = BATCH_SIZE
            update_every = UPDATE_EVERY
            
            buffer = ReplayBuffer(capacity=50000, device=device)
            best_val_sharpe = -float('inf')

            # ====================== 11 个数据集自适应参数重构 ======================
            starting_p1 = train_lob[0, 0]
            starting_p2 = train_lob[0, 2]
            starting_mid = (max(starting_p1, starting_p2) + min(starting_p1, starting_p2)) / 2.0

            if starting_mid <= 1.0:
                adaptive_capital = 1000.0
            else:
                adaptive_capital = starting_mid * config["max_inventory"] * 3.0

            config["action_scale"] = float(ACTION_SCALE)

            print(f"\n开始运行 【{args.mode.upper()}】 跑数循环...")
            print(f"   自适应资产价格: {starting_mid:.4f} | 物理报价 Tick Size: {current_tick_size}")
            print(f"   自适应做市宽度 (Action Scale): {config['action_scale']:.2f} Ticks")
            print(f"   自适应准备金 (Capital): {adaptive_capital:.2f} (破产红线: {0.05 * adaptive_capital:.2f})")
            print("-" * 60)
            # ======================================================================

            # 🎯 经典启发式报价极速度验证通道 (AS 与 Ho-Stoll 经典算法)
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
                    
                    action_mm_val = agent.get_action(curr_mid, val_env.inventory, current_tick_size)
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
                val_inv_variance = np.var(val_inventories)

                print(
                    f"==> 经典 {args.mode.upper()} 评估结束 | "
                    f"样本外日均夏普: {val_sharpe_real:.4f} | "
                    f"最大回撤: {val_max_dd * 100:.2f}% | "
                    f"成交次数: {val_env.trade_count} | "
                    f"仓位波动: {np.min(val_inventories):+.1f} ~ {np.max(val_inventories):+.1f} | "
                    f"库存方差: {val_inv_variance:.4f}"
                )
                
                df_traj = pd.DataFrame({
                    "step": range(len(portfolio_history)),
                    "portfolio_value": portfolio_history,
                    "inventory": val_inventories
                })
                df_traj.to_csv(f"trajectory_data_{args.mode}_{DATASET_TYPE}.csv", index=False)
                
                test_results[DATASET_TYPE] = f"✅ 学术跑数成功 ({args.mode.upper()} 极速完结, 最佳 Sharpe: {val_sharpe_real:.4f})"
                print("-" * 60)
                continue

            # ====================== 神经网络基线模型 (Neural Network Agents) 跑数循环 ======================
            # 🎯 将循环上限由 MAX_EPOCHS 改为局部命名空间安全的 epochs_to_run [3]
            for epoch in range(1, epochs_to_run + 1):
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

                while not done:
                    with torch.no_grad():
                        if args.mode == "our":
                            s_t, _, _ = agent.encoder(state)
                            action_mm, _ = agent.actor_mm(s_t)
                            action_it, _ = agent.actor_it(s_t)
                        elif args.mode == "mappo":
                            s_t = agent.encoder(state)
                            action_mm, _ = agent.actor_mm(s_t)
                            action_it = agent.actor_it(s_t) # 🎯 修正核心多智能体基线 Bug
                        elif args.mode in ["sac", "deeplob", "timesfm", "moment", "timer_xl"]:
                            s_t = agent.encoder(state)
                            action_mm, _ = agent.actor_mm(s_t)
                            action_it = torch.zeros(1, 1).to(device)
                        elif args.mode == "ppo":
                            s_t = agent.encoder(state)
                            action_mm, _ = agent.actor_mm(s_t)
                            action_it = torch.zeros(1, 1).to(device)
                        elif args.mode == "hrl":
                            s_t = agent.encoder(state)
                            goal, _ = agent.manager(s_t)
                            s_worker = torch.cat([s_t, goal], dim=-1)
                            action_mm, _ = agent.actor_mm(s_worker)
                            action_it = goal # 🎯 修复 HRL 交互层

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
                              f"Avg_Reward: {np.mean(epoch_rewards[-100:]):.6f}")

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
                    with torch.no_grad():
                        if args.mode == "our":
                            s_t_val, _, _ = agent.encoder(val_state)
                            action_mm_val, _ = agent.actor_mm(s_t_val, deterministic=True, with_logprob=False)
                            action_it_val, _ = agent.actor_it(s_t_val, deterministic=True, with_logprob=False)
                        elif args.mode in ["sac", "deeplob", "timesfm", "moment", "timer_xl"]:
                            s_t_val = agent.encoder(val_state)
                            action_mm_val, _ = agent.actor_mm(s_t_val, deterministic=True, with_logprob=False)
                            action_it_val = torch.zeros(1, 1).to(device)
                        elif args.mode == "ppo":
                            s_t_val = agent.encoder(val_state)
                            action_mm_val, _ = agent.actor_mm(s_t_val, deterministic=True, with_logprob=False)
                            action_it_val = torch.zeros(1, 1).to(device)
                        elif args.mode == "hrl":
                            s_t_val = agent.encoder(val_state)
                            goal_val, _ = agent.manager(s_t_val)
                            s_worker_val = torch.cat([s_t_val, goal_val], dim=-1)
                            action_mm_val, _ = agent.actor_mm(s_worker_val, deterministic=True, with_logprob=False)
                            action_it_val = goal_val # 已修复
                        elif args.mode == "mappo":
                            s_t_val = agent.encoder(val_state)
                            action_mm_val, _ = agent.actor_mm(s_t_val, deterministic=True, with_logprob=False)
                            action_it_val = agent.actor_it(s_t_val, deterministic=True, with_logprob=False) # 已修复

                    next_val_state, val_reward, val_done = val_env.step(action_mm_val, action_it_val)
                    val_inventories.append(val_env.inventory)

                    p1 = val_env.lob_data[val_env.current_tick - 1, 0]
                    p2 = val_env.lob_data[val_env.current_tick - 1, 2]
                    mid_val = (max(p1, p2) + min(p1, p2)) / 2.0
                    p_val = val_env.cash + val_env.inventory * mid_val
                    portfolio_history.append(p_val)

                    val_state = next_val_state

                val_sharpe_real = calculate_real_daily_sharpe(
                    portfolio_history=portfolio_history,
                    dataset_type=DATASET_TYPE,
                    bar_size=2000,
                    trade_count=val_env.trade_count,
                    initial_capital=adaptive_capital
                )

                sharpe_upper_bound = 100.0
                sharpe_lower_bound = -100.0
                if val_sharpe_real > sharpe_upper_bound:
                    print(f"   ⚠️ [过拟合警报] 样本外日均夏普达 {val_sharpe_real:.4f}，超出高频做市理论上限")
                elif val_sharpe_real < sharpe_lower_bound:
                    print(f"   ⚠️ [亏损硬预警] 样本外日均夏普达 {val_sharpe_real:.4f}，落入严重亏损区间")

                full_nav = np.array(portfolio_history)
                cum_max = np.maximum.accumulate(full_nav)
                denominator = np.maximum(cum_max, adaptive_capital)
                drawdowns = (cum_max - full_nav) / (denominator + 1e-12)

                val_max_dd = np.clip(np.max(drawdowns), 0.0, 1.0)
                val_inv_variance = np.var(val_inventories)

                print(
                    f"==> Epoch {epoch:2d} 结束 | "
                    f"样本外日均夏普: {val_sharpe_real:.4f} | "
                    f"最大回撤: {val_max_dd * 100:.2f}% | "
                    f"成交次数: {val_env.trade_count} | "
                    f"仓位波动: {np.min(val_inventories):+.1f} ~ {np.max(val_inventories):+.1f} | "
                    f"库存方差: {val_inv_variance:.4f}"
                )

                if val_sharpe_real > best_val_sharpe:
                    best_val_sharpe = val_sharpe_real
                    best_model_state = {
                        "encoder": agent.encoder.state_dict(),
                        "actor_mm": agent.actor_mm.state_dict(),
                        "epoch": epoch,
                        "val_sharpe": val_sharpe_real,
                        "val_max_dd": val_max_dd,
                        "val_inv_variance": val_inv_variance
                    }
                    if hasattr(agent, "manager"):
                        best_model_state["manager"] = agent.manager.state_dict()
                        
                    torch.save(best_model_state, f"best_hfmm_model_{args.mode}_{DATASET_TYPE}.pth")

                    df_traj = pd.DataFrame({
                        "step": range(len(portfolio_history)),
                        "portfolio_value": portfolio_history,
                        "inventory": val_inventories
                    })
                    df_traj.to_csv(f"trajectory_data_{args.mode}_{DATASET_TYPE}.csv", index=False)
                    print(f"   📊 [最佳轨迹已更新] 自动保存至: trajectory_data_{args.mode}_{DATASET_TYPE}.csv")

            test_results[DATASET_TYPE] = f"✅ 学术跑数成功 ({epochs_to_run} Epochs 完结, 最佳 Sharpe: {best_val_sharpe:.4f})"

        except Exception as e:
            test_results[DATASET_TYPE] = f"❌ 运行报错 (FAILED: {str(e)})"
            traceback.print_exc()

        print("-" * 60)

    print("\n" + "=" * 65)
    print(f"📊 【DIB-MASAC 实验跑数系统 - 状态结算面板】 | 运行模式: {args.mode.upper()}")
    print("=" * 65)
    for market, status in test_results.items():
        print(f"  * 目标细分市场: 【{market:22s}】 ===> 训练状态: {status}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()