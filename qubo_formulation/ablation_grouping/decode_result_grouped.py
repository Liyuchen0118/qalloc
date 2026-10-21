import json
import argparse
import sys
import numpy as np
import os
import math

from typing import List, Dict

# Copy evaluate function
def calculate_precision_loss(bit: int, original_bit_width: int, sensitivity: float, alpha: float) -> float:
    denominator = 1.0 - np.exp(-alpha)
    if np.isclose(denominator, 0): return np.inf
    numerator = np.exp(-alpha * (bit / original_bit_width)) - np.exp(-alpha)
    return sensitivity * (numerator / denominator)

def calculate_final_score(final_assigned_bits, sensitivities, layer_params, target_bit_widths, original_bit_width, alpha, target_avg_bit_width, w_acc=0.7, w_mem=0.3, strictness=2.0) -> Dict[str, float]:
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
    return {"Accuracy Score": score_acc, "Memory Score": score_mem, "Final Score": final_score, "Total Accuracy Loss": current_loss, "Final Average Bit-width": actual_avg_bit_width}

param_order = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"
]

def load_and_flatten_json(file_path: str) -> np.ndarray:
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="解码分组QUBO的分配结果")
    parser.add_argument("--mapping_path", type=str, default="group_mapping_dict.json", help="分组映射字典的路径")
    parser.add_argument("--solution_path", type=str, required=False, default=None, help="包含各组位宽分配的JSON列表路径 (如果为空，将使用代码中粘贴的数组)")
    parser.add_argument("--model_name", type=str, default="llama-30b", help="评估用的模型名称")
    args = parser.parse_args()
    
    # ==============================================================================
    # --- 步骤1: 将云平台返回的0/1解向量数组粘贴到这里 ---
    # ==============================================================================
    pasted_solution_vector =  [ 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 ]
    
    
    
    
    with open(args.mapping_path, "r", encoding="utf-8") as f:
        group_mapping_dict = json.load(f)
        
    if args.solution_path:
        with open(args.solution_path, "r", encoding="utf-8") as f:
            group_solutions = json.load(f)
    else:
        group_solutions = pasted_solution_vector
        if not group_solutions:
            print("错误: 请通过 --solution_path 传入 JSON 文件路径，或者直接将数组粘贴到 pasted_solution_vector 中！")
            sys.exit(1)
        
    NUM_GROUPS = len(group_mapping_dict)
    num_decision_vars = NUM_GROUPS * 2
    
    if len(group_solutions) < num_decision_vars:
        print(f"错误: 结果向量的长度({len(group_solutions)})小于预期的决策变量数({num_decision_vars})！")
        sys.exit(1)
        
    solution_vector = group_solutions[:num_decision_vars]
    
    final_assigned_bits = []
    for i in range(NUM_GROUPS):
        y0_val = solution_vector[i * 2]
        y1_val = solution_vector[i * 2 + 1]
        if y0_val == 0 and y1_val == 0:
            final_assigned_bits.append(2)
        elif y0_val == 0 and y1_val == 1:
            final_assigned_bits.append(3)
        elif y0_val == 1 and y1_val == 0:
            final_assigned_bits.append(4)
        else:
            final_assigned_bits.append(8)
            
    # 计算全部层数
    NUM_LAYERS = sum(len(indices) for indices in group_mapping_dict.values())
    decoded_allocation = [0] * NUM_LAYERS
    
    # 还原分配
    for i in range(NUM_GROUPS):
        group_key = f"group_{i}"
        bit_width = final_assigned_bits[i]
        for layer_idx in group_mapping_dict[group_key]:
            decoded_allocation[layer_idx] = bit_width
            
    print(f"成功将 {NUM_GROUPS} 个组的结果还原为 {NUM_LAYERS} 个独立层的位宽分配。")
    
    # 直接在终端输出位宽分配，英文逗号隔开，不换行
    print("\n--- 完整位宽分配 ---")
    print(", ".join(map(str, decoded_allocation)))
    print("--------------------\n")
    
    # 尝试评估结果
    possible_paths = [
        f"model_info/{args.model_name}",
        f"../model_info/{args.model_name}",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_info", args.model_name)
    ]
    
    MODEL_INFO_PATH = None
    layer_params_path = None
    fisher_path = None
    for p in possible_paths:
        if os.path.exists(os.path.join(p, "LayerParams.json")):
            MODEL_INFO_PATH = p
            layer_params_path = os.path.join(MODEL_INFO_PATH, "LayerParams.json")
            fisher_path = os.path.join(MODEL_INFO_PATH, "fisher.json")
            break

    if layer_params_path and os.path.exists(layer_params_path) and os.path.exists(fisher_path):
        try:
            layer_params = load_and_flatten_json(layer_params_path)
            sensitivities = load_and_flatten_json(fisher_path)
            
            TARGET_BIT_WIDTHS = [2, 3, 4, 8]
            ORIGINAL_BIT_WIDTH = 16
            ALPHA = 10.0
            COMPRESSION_RATIO = 0.25
            target_avg_bit_width = ORIGINAL_BIT_WIDTH * COMPRESSION_RATIO
            
            scores = calculate_final_score(
                decoded_allocation, sensitivities, layer_params,
                TARGET_BIT_WIDTHS, ORIGINAL_BIT_WIDTH, ALPHA, target_avg_bit_width
            )
            
            print("\n" + "="*50)
            print("           解码分配方案评估")
            print("="*50)
            print(f"最终总精度损失 (ΔAcc): {scores['Total Accuracy Loss']:.4f}")
            print(f"最终压缩率: {scores['Final Average Bit-width'] / ORIGINAL_BIT_WIDTH:.4f} (目标: {COMPRESSION_RATIO})")
            print(f"最终平均位宽: {scores['Final Average Bit-width']:.2f} bits (目标: {target_avg_bit_width:.2f} bits)")
            print("\n--- 百分制综合评估 ---")
            print(f"精度得分 (Score_Acc): {scores['Accuracy Score']:.2f} / 100")
            print(f"内存收敛得分 (Score_Mem): {scores['Memory Score']:.2f} / 100")
            print(f"最终综合评分 (Final Score): {scores['Final Score']:.2f} / 100")
            
        except Exception as e:
            print(f"\n评估过程出错: {e}")
    else:
        print(f"\n无法找到模型 {args.model_name} 的数据文件进行评估，跳过评估。")
