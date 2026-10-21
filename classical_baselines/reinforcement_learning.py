import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import os
import json
from typing import Tuple, List, Dict
import time

# ==============================================================================
# 模块一：数据加载与处理工具 (与所有baseline对齐)
# ==============================================================================
param_order = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"
]

def load_and_flatten_json(file_path: str) -> np.ndarray:
    """从指定路径加载JSON文件并将其值平铺成一维Numpy数组。"""
    with open(file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    top_level_keys = list(json_data.keys())
    if len(top_level_keys) == 1:
        data_to_process = json_data[top_level_keys[0]]
    else:
        data_to_process = json_data
    flat_data = []
    for layer_idx in sorted(data_to_process.keys(), key=int):
        layer_block = data_to_process[layer_idx]
        for param_name in param_order:
            if param_name in layer_block:
                flat_data.append(layer_block[param_name])
    return np.array(flat_data)

def same_layer_reset(data: np.ndarray) -> Dict[str, List[float]]:
    """将平铺的数据按层组名重新分组"""
    num_layers_in_block = len(data) // len(param_order)
    result = {param: [] for param in param_order}
    for i in range(num_layers_in_block):
        start_idx = i * len(param_order)
        layer_data = data[start_idx : start_idx + len(param_order)]
        for param, value in zip(param_order, layer_data):
            result[param].append(value)
    return result

def pack_list(data: Dict[str, List[int]]) -> List[int]:
    """将分组后的分配列表重新平铺成一个列表"""
    num_layers_in_group = len(data[param_order[0]])
    new_data = []
    for layer_idx in range(num_layers_in_group):
        for param in param_order:
            new_data.append(data[param][layer_idx])
    return new_data

# ==============================================================================
# 模块二：百分制评分指标 (与所有baseline对齐)
# ==============================================================================
def calculate_precision_loss(bit: int, original_bit_width: int, sensitivity: float, alpha: float) -> float:
    """严格按照论文草稿中的公式计算单层的精度损失 (ΔAcc)。"""
    denominator = 1.0 - np.exp(-alpha)
    if np.isclose(denominator, 0): return np.inf
    numerator = np.exp(-alpha * (bit / original_bit_width)) - np.exp(-alpha)
    return sensitivity * (numerator / denominator)

def calculate_final_score(
    final_assigned_bits: List[int],
    sensitivities: np.ndarray,
    layer_params: np.ndarray,
    target_bit_widths: List[int],
    original_bit_width: int,
    alpha: float,
    target_avg_bit_width: float,
    w_acc: float = 0.7,
    w_mem: float = 0.3
) -> Dict[str, float]:
    """
    计算一个给定分配方案的百分制综合评分。
    """
    num_layers = len(layer_params)

    # 1. 计算基准值
    loss_max = np.sum([
        calculate_precision_loss(min(target_bit_widths), original_bit_width, sensitivities[i], alpha)
        for i in range(num_layers)
    ])
    loss_min = np.sum([
        calculate_precision_loss(max(target_bit_widths), original_bit_width, sensitivities[i], alpha)
        for i in range(num_layers)
    ])
    
    # 2. 计算当前方案的精度损失
    current_loss = np.sum([
        calculate_precision_loss(final_assigned_bits[i], original_bit_width, sensitivities[i], alpha)
        for i in range(num_layers)
    ])

    # 3. 计算精度得分
    if (loss_max - loss_min) > 1e-6:
        normalized_loss = (current_loss - loss_min) / (loss_max - loss_min)
        score_acc = 100 * (1 - normalized_loss)
    else:
        score_acc = 100.0

    # 4. 计算当前方案的内存偏差
    actual_avg_bit_width = np.average(final_assigned_bits, weights=layer_params)
    delta_b = abs(actual_avg_bit_width - target_avg_bit_width)
    
    # 5. 计算内存收敛得分
    delta_b_max = max(
        abs(max(target_bit_widths) - target_avg_bit_width),
        abs(min(target_bit_widths) - target_avg_bit_width)
    )
    if delta_b_max > 1e-6:
        score_mem = 100 * max(0, 1 - (delta_b / delta_b_max))
    else:
        score_mem = 100.0

    # 6. 计算最终综合得分
    final_score = w_acc * score_acc + w_mem * score_mem

    return {
        "Accuracy Score": score_acc,
        "Memory Score": score_mem,
        "Final Score": final_score,
        "Total Accuracy Loss": current_loss,
        "Final Average Bit-width": actual_avg_bit_width
    }

# ==============================================================================
# 模块三：强化学习环境与智能体
# ==============================================================================
# 超参数
EPISODES = 2000
BATCH_SIZE = 64
GAMMA = 0.99
CLIP_EPSILON = 0.2
LR_ACTOR = 3e-4
LR_CRITIC = 1e-3
UPDATE_ITERS = 10
REWARD_PENALTY_WEIGHT = 100.0 # 奖励函数中对内存偏差的惩罚权重

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BitAllocationEnv:
    def __init__(self, layer_params, bits, sensitivities, alpha, target_avg_bit_width, origin_bit):
        self.p = torch.tensor(layer_params, dtype=torch.float32, device=device)
        self.bits = torch.tensor(bits, dtype=torch.float32, device=device)
        self.F = torch.tensor(sensitivities, dtype=torch.float32, device=device)
        self.alpha = torch.tensor(alpha, dtype=torch.float32, device=device)
        self.target_avg_bit_width = torch.tensor(target_avg_bit_width, dtype=torch.float32, device=device)
        self.origin_bit = torch.tensor(origin_bit, dtype=torch.float32, device=device)
        self.n_layers = len(sensitivities)
        
        # 修正：target_avg_bit_width 是一个 float，直接转换为 int
        self.best_allocations = [int(target_avg_bit_width)] * self.n_layers
        self.best_reward = self._calculate_reward_for_allocation(self.best_allocations)
        
        self.reset()

    def reset(self):
        self.current_layer = 0
        self.allocated_bits = []
        return self._get_state()

    def _get_state(self):
        # 状态表示：[当前层进度, 敏感度向量, 参数量向量]
        state = torch.zeros(1 + self.n_layers * 2, device=device)
        state[0] = self.current_layer / self.n_layers
        state[1:1+self.n_layers] = self.F / torch.max(self.F)
        state[1+self.n_layers:] = self.p / torch.max(self.p)
        return state

    def step(self, action):
        bit_value = self.bits[action]
        self.allocated_bits.append(bit_value.item())
        self.current_layer += 1
        
        done = self.current_layer >= self.n_layers
        
        if done:
            reward = self._calculate_final_reward()
            if reward.item() > self.best_reward:
                self.best_reward = reward.item()
                self.best_allocations = self.allocated_bits.copy()
        else:
            reward = torch.tensor(0.0, device=device) # 中间奖励为0
            
        next_state = self._get_state()
        return next_state, reward, done, {}

    def _calculate_reward_for_allocation(self, allocations):
        allocations_tensor = torch.tensor(allocations, dtype=torch.float32, device=device)
        
        # 1. 计算精度损失 (需要最小化，所以在奖励中取负)
        # 修正：使用标准化的精度损失计算函数
        loss_terms = [
            calculate_precision_loss(bit.item(), self.origin_bit.item(), sens.item(), self.alpha.item())
            for sens, bit in zip(self.F, allocations_tensor)
        ]
        accuracy_loss = torch.sum(torch.tensor(loss_terms, device=device))
        
        # 2. 计算内存约束惩罚 (需要最小化，所以在奖励中取负)
        # 修正：使用数值稳定的平均位宽
        actual_avg_bit_width = torch.sum(self.p * allocations_tensor) / torch.sum(self.p)
        budget_penalty = (actual_avg_bit_width - self.target_avg_bit_width)**2
        
        # 最终奖励 = - (精度损失 + 加权惩罚)
        return (-accuracy_loss - REWARD_PENALTY_WEIGHT * budget_penalty).item()
    
    def _calculate_final_reward(self):
        return torch.tensor(self._calculate_reward_for_allocation(self.allocated_bits), device=device)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU(),
                                 nn.Linear(256, action_dim), nn.Softmax(dim=-1))
        self.to(device)
    def forward(self, state): return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU(),
                                 nn.Linear(256, 1))
        self.to(device)
    def forward(self, state): return self.net(state)

class PPO:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        self.buffer = []

    def select_action(self, state):
        with torch.no_grad():
            probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def update(self):
        if not self.buffer: return
        
        states, actions, log_probs_old, rewards, dones = zip(*self.buffer)
        self.buffer.clear()

        states = torch.stack(states).to(device)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        log_probs_old = torch.stack(log_probs_old).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)
        
        # 计算 GAE
        with torch.no_grad():
            values = self.critic(states).squeeze()
            advantages = torch.zeros_like(rewards)
            last_adv = 0
            # 修正: 确保 next_values 也有正确的维度
            next_values = torch.cat((values[1:], torch.tensor([0.0], device=device))) if len(values) > 1 else torch.tensor([0.0], device=device)
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + GAMMA * next_values[t] * (1-dones[t]) - values[t]
                advantages[t] = last_adv = delta + GAMMA * 0.95 * (1-dones[t]) * last_adv
            returns = advantages + values

        for _ in range(UPDATE_ITERS):
            # Actor loss
            probs = self.actor(states)
            dist = torch.distributions.Categorical(probs)
            log_probs_new = dist.log_prob(actions)
            ratio = torch.exp(log_probs_new - log_probs_old)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            self.actor_optim.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optim.step()

            # Critic loss
            critic_loss = nn.MSELoss()(self.critic(states).squeeze(), returns)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

def train(bits, F, p, alpha, R, sameLayerReset, origin_bit):
    if sameLayerReset:
        layer_group_F = same_layer_reset(F)
        layer_group_p = same_layer_reset(p)
        allocations = {}
        for group_name in param_order:
            print(f"\n--- Training Group: {group_name} ---")
            group_F = np.array(layer_group_F[group_name])
            group_p = np.array(layer_group_p[group_name])
            allocations[group_name] = train_all_layer(bits, group_F, group_p, alpha, R, origin_bit)
        return pack_list(allocations)
    else:
        return train_all_layer(bits, F, p, alpha, R, origin_bit)

def train_all_layer(bits, F, p, alpha, R, origin_bit):
    target_avg_bit_width = origin_bit * R
    env = BitAllocationEnv(p, bits, F, alpha, target_avg_bit_width, origin_bit)
    state_dim = env._get_state().shape[0]
    action_dim = len(bits)
    agent = PPO(state_dim, action_dim)

    for episode in range(EPISODES):
        state = env.reset()
        done = False
        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.buffer.append((state, action, log_prob, reward.item(), done))
            state = next_state
        
        agent.update()
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{EPISODES}, Best Reward So Far: {env.best_reward:.4f}")
    
    return [int(b) for b in env.best_allocations]

# ==============================================================================
# 主程序
# ==============================================================================
if __name__ == "__main__":
    MODEL_NAME = "llama-30b"
    MODEL_INFO_PATH = f"model_info/{MODEL_NAME}"
    
    bits = [2, 3, 4, 8]
    B = 16 
    R = 0.25
    alpha = 10.0
    sameLayerReset = True # 是否按层组进行训练

    print(f"--- 正在从 '{MODEL_INFO_PATH}' 加载数据 ---")
    p = load_and_flatten_json(os.path.join(MODEL_INFO_PATH, "LayerParams.json"))
    F_raw = load_and_flatten_json(os.path.join(MODEL_INFO_PATH, "fisher.json"))
    
    assert len(p) == len(F_raw), "数据长度不匹配！"
    print(f"成功加载了 {len(p)} 个层的指标。")

    print(f"\n--- 开始运行强化学习 (PPO) ---")
    t_start = time.time()
    best_allocation = train(bits, F_raw, p, alpha, R, sameLayerReset, origin_bit=B)
    t_end = time.time()
    
    if best_allocation:
        print("\n" + "="*50)
        print("           强化学习最终结果")
        print("="*50)
        
        target_avg_bit_width = B * R
        scores = calculate_final_score(
            final_assigned_bits=best_allocation,
            sensitivities=F_raw,
            layer_params=p,
            target_bit_widths=bits,
            original_bit_width=B,
            alpha=alpha,
            target_avg_bit_width=target_avg_bit_width
        )
        
        final_compression_ratio = scores["Final Average Bit-width"] / B
        
        print(f"求解耗时: {t_end - t_start:.2f} 秒")
        
        print("\n--- 分配策略摘要 ---")
        unique_bits, counts = np.unique(best_allocation, return_counts=True)
        for bit, count in zip(unique_bits, counts):
            print(f"  - {bit}-bit 被分配给了 {count} 个层")

        print("\n--- 完整位宽分配列表 ---")
        print(best_allocation)

        print("\n--- 最终性能指标 ---")
        print(f"最终总精度损失 (ΔAcc): {scores['Total Accuracy Loss']:.4f}")
        print(f"最终压缩率: {final_compression_ratio:.4f} (目标: {R})")
        print(f"最终平均位宽: {scores['Final Average Bit-width']:.2f} bits (目标: {target_avg_bit_width:.2f} bits)")
        
        print("\n--- 百分制综合评估 ---")
        print(f"精度得分 (Score_Acc): {scores['Accuracy Score']:.2f} / 100")
        print(f"内存收敛得分 (Score_Mem): {scores['Memory Score']:.2f} / 100")
        print(f"最终综合评分 (Final Score): {scores['Final Score']:.2f} / 100")
    else:
        print("\n❌ 优化过程未能找到任何有效解。")
