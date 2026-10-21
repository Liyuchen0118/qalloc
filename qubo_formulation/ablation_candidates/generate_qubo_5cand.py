import kaiwu as kw
import numpy as np
import os
import json
from typing import Tuple, List
import time
import math

# 确保已安装pandas库，用于处理CSV文件
try:
    import pandas as pd
except ImportError:
    print("\n[!!!] 错误: 未找到 'pandas' 库。")
    print("请先通过 'pip install pandas' 安装。")
    exit()

# ==============================================================================
# 模块一：数据加载与处理
# ==============================================================================
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

# ==============================================================================
# 模块二：QUBO模型的核心函数
# ==============================================================================
def calculate_precision_loss(bit: int, original_bit_width: int, sensitivity: float, alpha: float) -> float:
    denominator = 1.0 - np.exp(-alpha)
    if np.isclose(denominator, 0): return np.inf
    numerator = np.exp(-alpha * (bit / original_bit_width)) - np.exp(-alpha)
    return sensitivity * (numerator / denominator)

# ==============================================================================
# 主程序
# ==============================================================================
if __name__ == "__main__":
    MODEL_NAME = "qwen2.5-7b" 
    MODEL_INFO_PATH = f"model_info/{MODEL_NAME}"
    
    print(f"--- 步骤1: 正在从 '{MODEL_INFO_PATH}' 加载预计算的层级指标 ---")
    
    layer_params_path = os.path.join(MODEL_INFO_PATH, "LayerParams.json")
    fisher_path = os.path.join(MODEL_INFO_PATH, "fisher.json")
    
    try:
        layer_params = load_and_flatten_json(layer_params_path)
        sensitivities = load_and_flatten_json(fisher_path)
    except FileNotFoundError:
        print(f"⚠️ 找不到数据文件，请检查 {MODEL_INFO_PATH} 目录。为演示流程，将生成随机数据。")
        NUM_LAYERS = 28 * 7  
        layer_params = np.random.rand(NUM_LAYERS) * 1e7
        sensitivities = np.random.rand(NUM_LAYERS) * 0.1
    else:
        NUM_LAYERS = len(layer_params)
        
    print(f"成功加载了 {NUM_LAYERS} 个层的指标。")

    # --- 步骤2: 构建QUBO模型 ({2, 3, 4, 5, 6} 代数降阶复用版) ---
    TARGET_BIT_WIDTHS = [2, 3, 4, 5, 6]
    ORIGINAL_BIT_WIDTH = 16
    ALPHA = 10.0
    
    # 压缩率 0.25，目标平均位宽: 16 * 0.25 = 4-bit
    COMPRESSION_RATIO = 0.25
    
    total_params = np.sum(layer_params, dtype=np.float64)
    layer_weights = layer_params / total_params
    target_avg_bit_width = ORIGINAL_BIT_WIDTH * COMPRESSION_RATIO

    print(f"\n候选位宽: {TARGET_BIT_WIDTHS}")
    print(f"目标平均位宽: {target_avg_bit_width:.2f} bits")

    precision_loss_matrix = np.zeros((NUM_LAYERS, len(TARGET_BIT_WIDTHS)))
    for i in range(NUM_LAYERS):
        for j, bit in enumerate(TARGET_BIT_WIDTHS):
            precision_loss_matrix[i, j] = calculate_precision_loss(
                bit, ORIGINAL_BIT_WIDTH, sensitivities[i], ALPHA
            )
            
    qubo_model = kw.qubo.QuboModel()
    
    # 每层依然分配 4 个变量：y0, y1, y2 以及辅助变量 z (约束 z = y0 * y2)
    y = kw.qubo.ndarray((NUM_LAYERS, 4), 'y', kw.qubo.Binary)

    all_terms = []
    CONSTRAINT_PENALTY_WEIGHT = 2000.0
    AUX_PENALTY_WEIGHT = 5000.0  

    for i in range(NUM_LAYERS):
        y0, y1, y2, z = y[i, 0], y[i, 1], y[i, 2], y[i, 3]
        
        # 提取 {2, 3, 4, 5, 6} 的损失值
        L2, L3, L4, L5, L6 = precision_loss_matrix[i]
        
        # 1. 精度损失多项式系数 (已代入 z=y0*y2 降阶)
        l0 = L2
        l1 = L3 - L2
        l2 = L4 - L2
        l3 = L5 - L2
        l12 = L6 - L3 - L4 + L2
        l13 = L6 - L3 - L5 + L2
        l23 = L6 - L4 - L5 + L2
        l123 = L3 + L4 + L5 - L2 - 2 * L6 
        
        layer_precision_loss = (
            l1 * y0 + l2 * y1 + l3 * y2 +
            l12 * y0 * y1 + l13 * z + l23 * y1 * y2 + l123 * z * y1 +
            l0
        )
        all_terms.append(layer_precision_loss)
        
        # 2. 内存约束二次展开
        wi = layer_weights[i]
        
        # 新位宽公式系数 (b = 2 + 1y0 + 2y1 + 3y2 + 1y0y1 + 0z - 1y1y2 - 2zy1)
        # K = 基础常数 * 权重 - 目标平均位宽 * 权重
        K = wi * 2.0 - target_avg_bit_width * wi
        c1 = 1.0 * wi
        c2 = 2.0 * wi
        c3 = 3.0 * wi
        c12 = 1.0 * wi
        c13 = 0.0 * wi      # 在当前多项式下，z 独立项系数为 0
        c23 = -1.0 * wi
        c123 = -2.0 * wi
        
        # 严格同类项合并展开式 (通用的 z 吸收律处理)
        const_coeff = K**2
        y0_coeff = c1**2 + 2*K*c1
        y1_coeff = c2**2 + 2*K*c2
        y2_coeff = c3**2 + 2*K*c3
        
        z_coeff = c13**2 + 2*K*c13 + 2*c1*c3 + 2*c1*c13 + 2*c3*c13
        y0y1_coeff = c12**2 + 2*K*c12 + 2*c1*c2 + 2*c1*c12 + 2*c2*c12
        y1y2_coeff = c23**2 + 2*K*c23 + 2*c2*c3 + 2*c2*c23 + 2*c3*c23
        
        zy1_coeff = (c123**2 + 2*K*c123 + 
                     2*c1*c23 + 2*c1*c123 + 
                     2*c2*c13 + 2*c2*c123 + 
                     2*c3*c12 + 2*c3*c123 + 
                     2*c12*c13 + 2*c12*c23 + 2*c12*c123 + 
                     2*c13*c23 + 2*c13*c123 + 
                     2*c23*c123)
        
        layer_memory_penalty = (
            y0_coeff * y0 + y1_coeff * y1 + y2_coeff * y2 + z_coeff * z +
            y0y1_coeff * y0 * y1 + y1y2_coeff * y1 * y2 + zy1_coeff * z * y1 +
            const_coeff
        )
        all_terms.append(CONSTRAINT_PENALTY_WEIGHT * layer_memory_penalty)
        
        # 3. 辅助变量 z 的 Rosenberg 惩罚项 (强制满足 z = y0 * y2)
        aux_penalty = 3 * z + y0 * y2 - 2 * y0 * z - 2 * y2 * z
        all_terms.append(AUX_PENALTY_WEIGHT * aux_penalty)

    final_objective = kw.qubo.quicksum(all_terms)
    qubo_model.set_objective(final_objective)

    print("\n--- 步骤2: QUBO模型 (2-6选项代数降阶版) 构建完成 ---")

    # --- 步骤3: 生成QUBO矩阵并导出 ---
    original_qubo_mat = qubo_model.get_qubo_matrix()
    
    print(f"\n原始QUBO矩阵数值范围: [{original_qubo_mat.min():.2f}, {original_qubo_mat.max():.2f}]")
    print(f"   总变量数: {original_qubo_mat.shape[0]}")
    if original_qubo_mat.shape[0] == NUM_LAYERS * 4:
        print("✅ 变量数正确，每层成功压缩到 4 个变量！")

    print("\n正在将QUBO矩阵调整到8-bit精度空间 [-128, 127]...")
    adjusted_qubo_mat = kw.qubo.adjust_qubo_matrix_precision(original_qubo_mat, bit_width=8)
    
    # 更改了输出文件名以匹配新的位宽
    output_csv_path = "qwen_algebraic_qubo_8bit_5opts_2to6.csv"
    pd.DataFrame(adjusted_qubo_mat).to_csv(output_csv_path, index=False, header=False)
    
    print(f"\n✅ 成功生成矩阵！文件已保存至: {os.path.abspath(output_csv_path)}")