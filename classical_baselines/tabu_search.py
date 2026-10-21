import numpy as np
import os
import json
from typing import Tuple, List, Dict
import time
import math
from collections import deque
import kaiwu as kw # Kaiwu SDK 仅用于方便地构建QUBO矩阵

# ==============================================================================
# 模块一：数据加载与处理工具 (与所有baseline对齐)
# ==============================================================================
param_order = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"
]

def load_and_flatten_json(file_path: str) -> np.ndarray:
    """
    从指定路径加载JSON文件并将其值平铺成一维Numpy数组。
    """
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
    w_mem: float = 0.3,
    strictness: float = 2.0
) -> Dict[str, float]:
    """
    计算一个给定分配方案的百分制综合评分。
    """
    num_layers = len(layer_params)
    loss_max = np.sum([calculate_precision_loss(min(target_bit_widths), original_bit_width, sensitivities[i], alpha) for i in range(num_layers)])
    loss_min = np.sum([calculate_precision_loss(max(target_bit_widths), original_bit_width, sensitivities[i], alpha) for i in range(num_layers)])
    current_loss = np.sum([calculate_precision_loss(final_assigned_bits[i], original_bit_width, sensitivities[i], alpha) for i in range(num_layers)])
    score_acc = 100.0 * (1 - (current_loss - loss_min) / (loss_max - loss_min))**strictness if (loss_max - loss_min) > 1e-6 else 100.0
    actual_avg_bit_width = np.average(final_assigned_bits, weights=layer_params)
    delta_b = abs(actual_avg_bit_width - target_avg_bit_width)
    delta_b_max = max(abs(max(target_bit_widths) - target_avg_bit_width), abs(min(target_bit_widths) - target_avg_bit_width))
    score_mem = 100 * max(0, 1 - (delta_b / delta_b_max)) if delta_b_max > 1e-6 else 100.0
    final_score = w_acc * score_acc + w_mem * score_mem
    return {
        "Accuracy Score": score_acc, "Memory Score": score_mem, "Final Score": final_score,
        "Total Accuracy Loss": current_loss, "Final Average Bit-width": actual_avg_bit_width
    }

# ==============================================================================
# 模块三：禁忌搜索求解器
# ==============================================================================
class TabuSearchSolver:
    def __init__(self, bits, p, F, B, R, alpha, max_iter=1000, tabu_size=50):
        self.bits = bits
        self.p = p
        self.F = F
        self.B = B
        self.R = R
        self.alpha = alpha
        self.max_iter = max_iter
        self.tabu_size = tabu_size
        self.N = len(p)
        
        self.target_avg_bit_width = self.B * self.R
        # 适应度函数中的惩罚权重
        self.penalty_weight = 10000.0

    def fitness(self, individual: np.ndarray) -> float:
        """评估个体的适应度 (成本)，值越小越好"""
        objective = sum(
            calculate_precision_loss(bit, self.B, sensitivity, self.alpha)
            for sensitivity, bit in zip(self.F, individual)
        )
        current_avg_bit_width = np.average(individual, weights=self.p)
        constraint_violation = (current_avg_bit_width - self.target_avg_bit_width)**2
        return objective + self.penalty_weight * constraint_violation

    def run(self) -> np.ndarray:
        """运行禁忌搜索算法"""
        # 1. 初始化
        # 生成一个随机的初始解
        current_solution = np.random.choice(self.bits, size=self.N)
        best_solution = np.copy(current_solution)
        best_fitness = self.fitness(best_solution)
        
        tabu_list = deque(maxlen=self.tabu_size)
        
        # 2. 主循环
        for i in range(self.max_iter):
            best_neighbor = None
            best_neighbor_fitness = float('inf')
            best_neighbor_move = None

            # 3. 邻域搜索
            # 遍历所有可能的“单点移动”
            for layer_idx in range(self.N):
                original_bit = current_solution[layer_idx]
                for new_bit in self.bits:
                    if new_bit == original_bit:
                        continue
                    
                    # 创建一个邻居解
                    neighbor = np.copy(current_solution)
                    neighbor[layer_idx] = new_bit
                    
                    move = (layer_idx, new_bit)
                    neighbor_fitness = self.fitness(neighbor)
                    
                    # 4. 禁忌与特赦准则
                    is_tabu = move in tabu_list
                    
                    if not is_tabu:
                        if neighbor_fitness < best_neighbor_fitness:
                            best_neighbor = neighbor
                            best_neighbor_fitness = neighbor_fitness
                            best_neighbor_move = move
                    elif neighbor_fitness < best_fitness: # 特赦准则
                        best_neighbor = neighbor
                        best_neighbor_fitness = neighbor_fitness
                        best_neighbor_move = move
                        break # 找到特赦解，立即移动
                if best_neighbor_move and best_neighbor_fitness < best_fitness:
                    break
            
            if best_neighbor is None:
                # 如果没有找到任何可移动的邻居（不太可能发生）
                break

            # 5. 移动到最佳邻居
            current_solution = best_neighbor
            current_fitness = best_neighbor_fitness
            
            # 更新禁忌列表
            tabu_list.append(best_neighbor_move)
            
            # 更新历史最优解
            if current_fitness < best_fitness:
                best_solution = np.copy(current_solution)
                best_fitness = current_fitness
            
            if (i + 1) % 500 == 0:
                print(f"  [进度] 迭代 {i+1}/{self.max_iter}, 当前找到的最优成本: {best_fitness:.4f}")

        return best_solution

# ==============================================================================
# 主程序
# ==============================================================================
if __name__ == "__main__":
    # --- 步骤1: 定义实验参数 ---
    MODEL_NAME = "llama-30b"
    MODEL_INFO_PATH = f"model_info/{MODEL_NAME}"
    
    bits = [2, 3, 4, 8]  
    B = 16 
    R = 0.25
    alpha = 10.0
    
    # 禁忌搜索超参数
    MAX_ITERATIONS = 1000

    # --- 步骤2: 加载数据 ---
    print(f"--- 正在从 '{MODEL_INFO_PATH}' 加载数据 ---")
    p = load_and_flatten_json(os.path.join(MODEL_INFO_PATH, "LayerParams.json"))
    F_raw = load_and_flatten_json(os.path.join(MODEL_INFO_PATH, "fisher.json"))
    
    assert len(p) == len(F_raw), "数据长度不匹配！"
    print(f"成功加载了 {len(p)} 个层的指标。")

    # --- 步骤3: 运行禁忌搜索 ---
    print(f"\n--- 开始运行禁忌搜索 ({MAX_ITERATIONS}轮) ---")
    t_start = time.time()
    ts_solver = TabuSearchSolver(bits, p, F_raw, B, R, alpha, max_iter=MAX_ITERATIONS)
    best_allocation = ts_solver.run()
    t_end = time.time()
    
    # --- 步骤4: 输出结果与评估 ---
    if best_allocation is not None:
        print("\n" + "="*50)
        print("           禁忌搜索最终结果")
        print("="*50)
        
        target_avg_bit_width = B * R
        scores = calculate_final_score(
            final_assigned_bits=best_allocation.tolist(),
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
        print(best_allocation.tolist())

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
