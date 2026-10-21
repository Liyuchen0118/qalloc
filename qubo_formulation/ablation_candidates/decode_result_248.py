import numpy as np
import os
import json
from typing import Tuple, List, Dict
import math

# ==============================================================================
# 模块一：数据加载与处理
# ==============================================================================
param_order = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"
]

def load_and_flatten_json_with_names(file_path: str) -> Tuple[np.ndarray, List[str]]:
    """
    从指定路径加载JSON文件，将其值平铺成一维Numpy数组，并返回对应的层名称列表。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    top_level_keys = list(json_data.keys())
    if len(top_level_keys) == 1:
        data_to_process = json_data[top_level_keys[0]]
    else:
        data_to_process = json_data
    
    flat_data = []
    flat_names = []
    for layer_idx in sorted(data_to_process.keys(), key=int):
        layer_block = data_to_process[layer_idx]
        for param_name in param_order:
            if param_name in layer_block:
                flat_data.append(layer_block[param_name])
                full_name = f"layers.{layer_idx}.{param_name}"
                flat_names.append(full_name)
    return np.array(flat_data), flat_names

def calculate_precision_loss(bit: int, original_bit_width: int, sensitivity: float, alpha: float) -> float:
    """严格按照论文草稿中的公式计算单层的精度损失 (ΔAcc)。"""
    denominator = 1.0 - np.exp(-alpha)
    if np.isclose(denominator, 0): return np.inf
    numerator = np.exp(-alpha * (bit / original_bit_width)) - np.exp(-alpha)
    return sensitivity * (numerator / denominator)

# ==============================================================================
# 新增模块：百分制评分指标
# ==============================================================================
def calculate_final_score(
    final_assigned_bits: List[int],
    sensitivities: np.ndarray,
    layer_params: np.ndarray,
    target_bit_widths: List[int],
    original_bit_width: int,
    alpha: float,
    target_avg_bit_width: float,
    w_acc: float = 0.5,
    w_mem: float = 0.5
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
# 主程序：解读结果
# ==============================================================================
if __name__ == "__main__":
    # --- 步骤1: 预置了你从 CPQC-550 跑出的 {2, 4, 8} 最佳解向量 (-13576) ---
    full_solution_vector = [ 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0 ]
    
    # --- 步骤2: 重新生成与求解时一致的上下文数据 ---
    MODEL_NAME = "qwen2.5-7b" 
    MODEL_INFO_PATH = f"model_info/{MODEL_NAME}"
    
    print(f"--- 重新加载 '{MODEL_NAME}' 的上下文数据以进行结果分析 ---")
    
    layer_params_path = os.path.join(MODEL_INFO_PATH, "LayerParams.json")
    fisher_path = os.path.join(MODEL_INFO_PATH, "fisher.json")
    
    # 容错处理：为了确保代码可直接运行，如果本地没文件则生成模拟数据
    try:
        layer_params, layer_names = load_and_flatten_json_with_names(layer_params_path)
        sensitivities, _ = load_and_flatten_json_with_names(fisher_path)
    except FileNotFoundError:
        print(f"⚠️ 找不到数据文件，请检查 {MODEL_INFO_PATH} 目录。为演示流程，将生成随机数据。")
        NUM_LAYERS = len(full_solution_vector) // 2
        layer_params = np.random.rand(NUM_LAYERS) * 1e7
        sensitivities = np.random.rand(NUM_LAYERS) * 0.1
        layer_names = [f"layers.{i//7}.proj_{i%7}" for i in range(NUM_LAYERS)]
    else:
        NUM_LAYERS = len(layer_params)

    # 核心改动：对齐 {2, 4, 8} 版本
    TARGET_BIT_WIDTHS = [2, 4, 8]
    ORIGINAL_BIT_WIDTH = 16 
    ALPHA = 10.0
    COMPRESSION_RATIO = 0.25 # 目标平均 4-bit
    
    # --- 步骤3: 解读结果并进行评估 ---
    num_decision_vars = NUM_LAYERS * 2
    if len(full_solution_vector) < num_decision_vars:
        print(f"错误：结果向量的长度({len(full_solution_vector)})小于预期的决策变量数({num_decision_vars})！")
    else:
        solution_vector = full_solution_vector[:num_decision_vars]

        final_assigned_bits = []
        for i in range(NUM_LAYERS):
            y0_val = solution_vector[i * 2]
            y1_val = solution_vector[i * 2 + 1]
            
            # 核心改动：严格匹配 b = 2 + 2y0 + 6y1 - 2y0y1 映射
            if y0_val == 0 and y1_val == 0: 
                final_assigned_bits.append(2)
            elif y0_val == 1 and y1_val == 0: 
                final_assigned_bits.append(4)
            elif y0_val == 0 and y1_val == 1: 
                final_assigned_bits.append(8)
            else: 
                # y0_val == 1 and y1_val == 1 (状态复用映射为 8-bit)
                final_assigned_bits.append(8)

        # --- 详细方案输出 ---
        print("\n" + "="*50)
        print("           最终量化分配方案 (云平台结果)")
        print("="*50)
        print(f"{'层名称':<45} {'分配位宽':<12}")
        print("-" * 60)
        for i in range(NUM_LAYERS):
            print(f"{layer_names[i]:<45} {f'{final_assigned_bits[i]}-bit':<12}")
        print("-" * 60)
        
        # --- 调用评分函数进行评估 ---
        target_avg_bit_width = ORIGINAL_BIT_WIDTH * COMPRESSION_RATIO
        scores = calculate_final_score(
            final_assigned_bits=final_assigned_bits,
            sensitivities=sensitivities,
            layer_params=layer_params,
            target_bit_widths=TARGET_BIT_WIDTHS,
            original_bit_width=ORIGINAL_BIT_WIDTH,
            alpha=ALPHA,
            target_avg_bit_width=target_avg_bit_width,
            w_acc=0.7, 
            w_mem=0.3
        )
        
        # --- 最终性能指标与评分 ---
        unique_bits, counts = np.unique(final_assigned_bits, return_counts=True)
        print("\n--- 分配策略摘要 ---")
        for bit, count in zip(unique_bits, counts):
            print(f"  - {bit}-bit 被分配给了 {count} 个层")

        print("\n--- 完整位宽分配列表 ---")
        print(final_assigned_bits)

        print("\n--- 最终性能指标 ---")
        print(f"最终总精度损失 (ΔAcc): {scores['Total Accuracy Loss']:.4f}")
        print(f"最终平均位宽: {scores['Final Average Bit-width']:.2f} bits (目标: {target_avg_bit_width:.2f} bits)")
        
        print("\n--- 百分制综合评估 ---")
        print(f"精度得分 (Score_Acc): {scores['Accuracy Score']:.2f} / 100")
        print(f"内存收敛得分 (Score_Mem): {scores['Memory Score']:.2f} / 100")
        print(f"最终综合评分 (Final Score): {scores['Final Score']:.2f} / 100")