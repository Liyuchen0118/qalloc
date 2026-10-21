import numpy as np
import os
import json
from typing import Tuple, List, Dict
import time
import math
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
# 模块三：自建模拟退火求解器 (智能邻域搜索)
# ==============================================================================
def calculate_qubo_value(qubo_matrix: np.ndarray, solution_vector: np.ndarray) -> float:
    """计算给定解向量的QUBO目标函数值 E = x^T * Q * x"""
    return solution_vector.T @ qubo_matrix @ solution_vector

def custom_simulated_annealing_solver(
    qubo_matrix: np.ndarray,
    num_layers: int,
    num_levels: int,
    initial_temperature: float = 1000.0,
    alpha: float = 0.998,
    cutoff_temperature: float = 0.01,
    iterations_per_t: int = 1000,
    rand_seed: int = 42
) -> Tuple[np.ndarray, float]:
    np.random.seed(rand_seed)
    current_choices = np.random.randint(num_levels, size=num_layers)
    current_solution_vector = np.zeros(num_layers * num_levels, dtype=int)
    for i in range(num_layers):
        current_solution_vector[i * num_levels + current_choices[i]] = 1
    current_energy = calculate_qubo_value(qubo_matrix, current_solution_vector)
    best_solution_vector = np.copy(current_solution_vector)
    best_energy = current_energy
    current_temperature = initial_temperature
    temp_step_counter = 0
    while current_temperature > cutoff_temperature:
        for _ in range(iterations_per_t):
            layer_to_change = np.random.randint(num_layers)
            current_level = current_choices[layer_to_change]
            possible_new_levels = [l for l in range(num_levels) if l != current_level]
            if not possible_new_levels: continue
            new_level = np.random.choice(possible_new_levels)
            neighbor_solution_vector = np.copy(current_solution_vector)
            neighbor_solution_vector[layer_to_change * num_levels + current_level] = 0
            neighbor_solution_vector[layer_to_change * num_levels + new_level] = 1
            neighbor_energy = calculate_qubo_value(qubo_matrix, neighbor_solution_vector)
            delta_energy = neighbor_energy - current_energy
            if delta_energy < 0 or np.random.rand() < math.exp(-delta_energy / current_temperature):
                current_solution_vector = neighbor_solution_vector
                current_choices[layer_to_change] = new_level
                current_energy = neighbor_energy
                if current_energy < best_energy:
                    best_solution_vector = np.copy(current_solution_vector)
                    best_energy = current_energy
        current_temperature *= alpha
        temp_step_counter += 1
        if temp_step_counter % 500 == 0:
            print(f"  [进度] 温度已降至: {current_temperature:.2f}, 当前找到的最优能量值: {best_energy:.2f}")
    return best_solution_vector, best_energy

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

    # --- 步骤2: 加载数据 ---
    print(f"--- 正在从 '{MODEL_INFO_PATH}' 加载数据 ---")


    layer_params_file =  "LayerParams.json"
    fisher_file =  "fisher.json"

    p = load_and_flatten_json(os.path.join(MODEL_INFO_PATH, layer_params_file))
    F_raw = load_and_flatten_json(os.path.join(MODEL_INFO_PATH, fisher_file))
    
    assert len(p) == len(F_raw), "数据长度不匹配！"
    print(f"成功加载了 {len(p)} 个层的指标。")

    # --- 步骤3: 构建QUBO模型 ---
    print("\n--- 正在构建QUBO模型 ---")
    
    NUM_LAYERS = len(p)
    NUM_LEVELS = len(bits)
    
    total_params = np.sum(p, dtype=np.float64)
    layer_weights = p / total_params
    target_avg_bit_width = B * R

    precision_loss_matrix = np.zeros((NUM_LAYERS, NUM_LEVELS))
    for i in range(NUM_LAYERS):
        for j, bit in enumerate(bits):
            precision_loss_matrix[i, j] = calculate_precision_loss(bit, B, F_raw[i], alpha)
            
    qubo_model = kw.qubo.QuboModel()
    x = kw.qubo.ndarray((NUM_LAYERS, NUM_LEVELS), 'x', kw.qubo.Binary)
    
    objective_precision = kw.qubo.quicksum([x[i, j] * precision_loss_matrix[i, j] for i in range(NUM_LAYERS) for j in range(NUM_LEVELS)])
    actual_avg_bit_width_expr = kw.qubo.quicksum([x[i, j] * layer_weights[i] * bits[j] for i in range(NUM_LAYERS) for j in range(NUM_LEVELS)])
    memory_constraint_penalty = (actual_avg_bit_width_expr - target_avg_bit_width)**2
    
    CONSTRAINT_PENALTY_WEIGHT = 2000.0 
    final_objective = objective_precision + CONSTRAINT_PENALTY_WEIGHT * memory_constraint_penalty
    qubo_model.set_objective(final_objective)

    for i in range(NUM_LAYERS):
        one_choice_per_layer_expr = kw.qubo.quicksum([x[i, j] for j in range(NUM_LEVELS)])
        qubo_model.add_constraint(one_choice_per_layer_expr == 1, name=f"layer_{i}_choice", penalty=50000.0)

    qubo_mat = qubo_model.get_qubo_matrix()
    print("QUBO模型构建完成。")

    # --- 步骤4: 运行模拟退火求解器 ---
    print(f"\n--- 开始运行模拟退火求解器 ---")
    t_start = time.time()
    best_allocation_vector, qubo_val = custom_simulated_annealing_solver(
        qubo_matrix=qubo_mat,
        num_layers=NUM_LAYERS,
        num_levels=NUM_LEVELS,
        initial_temperature=5000,
        alpha=0.999,
        cutoff_temperature=0.01,
        iterations_per_t=NUM_LAYERS * 10,
        rand_seed=42
    )
    t_end = time.time()
    
    # --- 步骤5: 输出结果与评估 ---
    if qubo_val != float('inf'):
        print("\n" + "="*50)
        print("           模拟退火最终结果")
        print("="*50)
        
        # 解码独热编码
        solution_matrix = best_allocation_vector.reshape((NUM_LAYERS, NUM_LEVELS))
        chosen_level_indices = np.argmax(solution_matrix, axis=1)
        best_allocation = [bits[idx] for idx in chosen_level_indices]

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
        print("\n❌ 求解器未能找到有效解。")
