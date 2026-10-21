import kaiwu as kw
import numpy as np
import os
import json
import argparse

try:
    import pandas as pd
except ImportError:
    print("\n[!!!] 错误: 未找到 'pandas' 库。")
    print("请先通过 'pip install pandas' 安装。")
    exit()

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

def calculate_precision_loss(bit: int, original_bit_width: int, sensitivity: float, alpha: float) -> float:
    denominator = 1.0 - np.exp(-alpha)
    if np.isclose(denominator, 0): return np.inf
    numerator = np.exp(-alpha * (bit / original_bit_width)) - np.exp(-alpha)
    return sensitivity * (numerator / denominator)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成基于敏感度分组的QUBO矩阵")
    parser.add_argument("--model_name", type=str, default="llama-30b", help="模型名称")
    parser.add_argument("--group_size", type=int, default=4, help="每个组的大小")
    args = parser.parse_args()

    MODEL_NAME = args.model_name
    GROUP_SIZE = args.group_size
    
    # 找寻 model_info 的正确路径
    possible_paths = [
        f"model_info/{MODEL_NAME}",
        f"../model_info/{MODEL_NAME}",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_info", MODEL_NAME)
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
            
    if not MODEL_INFO_PATH:
        print(f"错误: 找不到模型 {MODEL_NAME} 的数据，请检查 model_info 目录。")
        exit(1)

    print(f"--- 步骤1: 正在加载 {MODEL_NAME} 数据并进行分组 ---")
    
    layer_params = load_and_flatten_json(layer_params_path)
    sensitivities = load_and_flatten_json(fisher_path)
    NUM_LAYERS = len(layer_params)
    
    # 获取根据敏感度排序的索引
    sorted_indices = np.argsort(sensitivities)
    
    # 根据 group_size 将层分为多个组
    groups = []
    for i in range(0, NUM_LAYERS, GROUP_SIZE):
        groups.append(sorted_indices[i:i + GROUP_SIZE].tolist())
    
    NUM_GROUPS = len(groups)
    print(f"成功加载 {NUM_LAYERS} 层，按敏感度分成了 {NUM_GROUPS} 个组 (每组最多 {GROUP_SIZE} 个)。")

    group_mapping_dict = {f"group_{i}": group for i, group in enumerate(groups)}
    
    mapping_output_path = "group_mapping_dict.json"
    with open(mapping_output_path, "w", encoding="utf-8") as f:
        json.dump(group_mapping_dict, f, indent=4)
        
    print(f"分组映射字典已保存至: {os.path.abspath(mapping_output_path)}")

    # --- 步骤2: 构建QUBO模型 ---
    TARGET_BIT_WIDTHS = [2, 3, 4, 8]
    ORIGINAL_BIT_WIDTH = 16
    ALPHA = 10.0
    COMPRESSION_RATIO = 0.25
    
    total_params = np.sum(layer_params, dtype=np.float64)
    layer_weights = layer_params / total_params
    target_avg_bit_width = ORIGINAL_BIT_WIDTH * COMPRESSION_RATIO

    # 计算组级别的权重和精度损失矩阵
    group_weights = np.zeros(NUM_GROUPS)
    precision_loss_matrix = np.zeros((NUM_GROUPS, len(TARGET_BIT_WIDTHS)))

    for g_idx, group in enumerate(groups):
        # 组的权重等于组内所有层权重的总和
        for layer_idx in group:
            group_weights[g_idx] += layer_weights[layer_idx]
            # 组在某个bit下的精度损失，等于该组内所有层在该bit下损失的总和
            for j, bit in enumerate(TARGET_BIT_WIDTHS):
                precision_loss_matrix[g_idx, j] += calculate_precision_loss(
                    bit, ORIGINAL_BIT_WIDTH, sensitivities[layer_idx], ALPHA
                )
                
    qubo_model = kw.qubo.QuboModel()
    y = kw.qubo.ndarray((NUM_GROUPS, 2), 'y', kw.qubo.Binary)

    all_terms = []
    CONSTRAINT_PENALTY_WEIGHT = 2000.0

    for i in range(NUM_GROUPS):
        y0, y1 = y[i, 0], y[i, 1]
        loss_2, loss_3, loss_4, loss_8 = precision_loss_matrix[i]
        
        layer_precision_loss = (
            (loss_4 - loss_2) * y0 +
            (loss_3 - loss_2) * y1 +
            (loss_2 - loss_3 - loss_4 + loss_8) * y0 * y1 +
            loss_2 
        )
        all_terms.append(layer_precision_loss)
        
        wi = group_weights[i]
        target = target_avg_bit_width
        b_const, b_y0, b_y1, b_y0y1 = 2, 2, 1, 3
        A = wi * (b_const - target)
        B = wi * b_y0
        C = wi * b_y1
        D = wi * b_y0y1
        const_coeff = A**2
        y0_coeff = B**2 + 2*A*B
        y1_coeff = C**2 + 2*A*C
        y0y1_coeff = D**2 + 2*A*D + 2*B*C + 2*B*D + 2*C*D
        layer_memory_penalty = (
            y0_coeff * y0 +
            y1_coeff * y1 +
            y0y1_coeff * y0 * y1 +
            const_coeff
        )
        all_terms.append(CONSTRAINT_PENALTY_WEIGHT * layer_memory_penalty)

    final_objective = kw.qubo.quicksum(all_terms)
    qubo_model.set_objective(final_objective)

    print("\n--- 步骤2: 基于分组的 QUBO模型构建完成 ---")

    # --- 步骤3: 生成QUBO矩阵并导出为CSV文件 ---
    original_qubo_mat = qubo_model.get_qubo_matrix()
    
    print(f"\n原始QUBO矩阵数值范围: [{original_qubo_mat.min():.2f}, {original_qubo_mat.max():.2f}]")
    print(f"   总变量数 (预期为 {NUM_GROUPS} * 2): {original_qubo_mat.shape[0]}")
    
    print("\n正在将QUBO矩阵调整到8-bit精度空间 [-128, 127]...")
    adjusted_qubo_mat = kw.qubo.adjust_qubo_matrix_precision(original_qubo_mat, bit_width=8)
    
    output_csv_path = f"{MODEL_NAME}_grouped_size{GROUP_SIZE}_qubo_8bit.csv"
    pd.DataFrame(adjusted_qubo_mat).to_csv(output_csv_path, index=False, header=False)
    
    print(f"\n✅ 成功生成符合硬件要求的8-bit QUBO矩阵！")
    print(f"   文件已保存至: {os.path.abspath(output_csv_path)}")
    print("\n下一步：请登录云平台上传此CSV文件进行计算。")
