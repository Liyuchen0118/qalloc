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
# 模块一：数据加载与处理 (无修改)
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

# ==============================================================================
# 模块二：QUBO模型的核心函数 (无修改)
# ==============================================================================
def calculate_precision_loss(bit: int, original_bit_width: int, sensitivity: float, alpha: float) -> float:
    """严格按照论文草稿中的公式计算单层的精度损失 (ΔAcc)。"""
    denominator = 1.0 - np.exp(-alpha)
    if np.isclose(denominator, 0): return np.inf
    numerator = np.exp(-alpha * (bit / original_bit_width)) - np.exp(-alpha)
    return sensitivity * (numerator / denominator)

# ==============================================================================
# 主程序
# ==============================================================================
if __name__ == "__main__":
    # --- 步骤1: 从JSON文件加载模型数据 ---
    MODEL_NAME = "qwen2.5-7b" 
    MODEL_INFO_PATH = f"model_info/{MODEL_NAME}"
    
    print(f"--- 步骤1: 正在从 '{MODEL_INFO_PATH}' 加载预计算的层级指标 ---")
    
    layer_params_path = os.path.join(MODEL_INFO_PATH, "LayerParams.json")
    fisher_path = os.path.join(MODEL_INFO_PATH, "fisher.json")
    
    # 注意：运行前需确保目录和文件存在
    try:
        layer_params = load_and_flatten_json(layer_params_path)
        sensitivities = load_and_flatten_json(fisher_path)
    except FileNotFoundError:
        print(f"⚠️ 找不到数据文件，请检查 {MODEL_INFO_PATH} 目录。为演示流程，将生成随机数据。")
        NUM_LAYERS = 28 * 7  # 假设28层，每层7个矩阵
        layer_params = np.random.rand(NUM_LAYERS) * 1e7
        sensitivities = np.random.rand(NUM_LAYERS) * 0.1
    else:
        NUM_LAYERS = len(layer_params)
        
    print(f"成功加载了 {NUM_LAYERS} 个层的指标。")

    # --- 步骤2: 构建QUBO模型 ({2, 4, 8} 3选项版本) ---
    # 核心修改：候选位宽更新为 3 种
    TARGET_BIT_WIDTHS = [2, 4, 8]
    ORIGINAL_BIT_WIDTH = 16
    ALPHA = 10.0
    COMPRESSION_RATIO = 0.25 # 目标平均 4-bit
    
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
    # 仍然是每层 2 个变量，没有引入任何辅助变量
    y = kw.qubo.ndarray((NUM_LAYERS, 2), 'y', kw.qubo.Binary)

    all_terms = []
    CONSTRAINT_PENALTY_WEIGHT = 2000.0

    for i in range(NUM_LAYERS):
        y0, y1 = y[i, 0], y[i, 1]
        loss_2, loss_4, loss_8 = precision_loss_matrix[i]
        
        # 1. 精度损失多项式 (状态复用映射)
        # 状态: (0,0)->2, (1,0)->4, (0,1)->8, (1,1)->8
        layer_precision_loss = (
            (loss_4 - loss_2) * y0 +
            (loss_8 - loss_2) * y1 +
            (loss_2 - loss_4) * y0 * y1 +
            loss_2 
        )
        all_terms.append(layer_precision_loss)
        
        # 2. 内存约束代数惩罚项
        # 基础位宽公式: b = 2 + 2*y0 + 6*y1 - 2*y0*y1
        wi = layer_weights[i]
        target = target_avg_bit_width
        b_const, b_y0, b_y1, b_y0y1 = 2, 2, 6, -2  # 核心系数已替换
        
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

    print("\n--- 步骤2: QUBO模型 ({2, 4, 8} 版本) 构建完成 ---")

    # --- 步骤3: 生成QUBO矩阵，进行精度适配，并导出为CSV文件 ---
    original_qubo_mat = qubo_model.get_qubo_matrix()
    
    print(f"\n原始QUBO矩阵数值范围: [{original_qubo_mat.min():.2f}, {original_qubo_mat.max():.2f}]")
    print(f"   总变量数: {original_qubo_mat.shape[0]}")
    if original_qubo_mat.shape[0] == NUM_LAYERS * 2:
        print("✅ 变量数正确，成功保持了 2x 倍数架构！")
    else:
        print("⚠️ 变量数多于预期，请检查维度。")

    print("\n正在将QUBO矩阵调整到8-bit精度空间 [-128, 127]...")
    adjusted_qubo_mat = kw.qubo.adjust_qubo_matrix_precision(original_qubo_mat, bit_width=8)
    print(f"调整后QUBO矩阵数值范围: [{adjusted_qubo_mat.min()}, {adjusted_qubo_mat.max()}]")
    
    output_csv_path = "qwen_algebraic_qubo_8bit_3opts.csv"
    
    pd.DataFrame(adjusted_qubo_mat).to_csv(output_csv_path, index=False, header=False)
    
    print(f"\n✅ 成功生成符合硬件要求的8-bit QUBO矩阵！")
    print(f"   文件已保存至: {os.path.abspath(output_csv_path)}")