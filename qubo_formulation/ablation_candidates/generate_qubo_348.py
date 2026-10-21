
import kaiwu as kw
import numpy as np
import os
import json
import pandas as pd

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
    data_to_process = json_data[top_level_keys[0]] if len(top_level_keys) == 1 else json_data
    
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

# ==============================================================================
# 主程序
# ==============================================================================
if __name__ == "__main__":
    MODEL_NAME = "llama-7b" 
    MODEL_INFO_PATH = f"model_info/{MODEL_NAME}"
    
    try:
        layer_params = load_and_flatten_json(os.path.join(MODEL_INFO_PATH, "LayerParams.json"))
        sensitivities = load_and_flatten_json(os.path.join(MODEL_INFO_PATH, "fisher.json"))
    except FileNotFoundError:
        NUM_LAYERS = 196
        layer_params = np.random.rand(NUM_LAYERS) * 1e7
        sensitivities = np.random.rand(NUM_LAYERS) * 0.1
    else:
        NUM_LAYERS = len(layer_params)
        
    print(f"成功加载了 {NUM_LAYERS} 个层的指标。")

    # --- 核心修改：候选位宽更新为 {3, 4, 8} ---
    TARGET_BIT_WIDTHS = [3, 4, 8]
    ORIGINAL_BIT_WIDTH = 16
    ALPHA = 7.0
    COMPRESSION_RATIO = 0.25 # 目标平均 4-bit
    
    total_params = np.sum(layer_params, dtype=np.float64)
    layer_weights = layer_params / total_params
    target_avg_bit_width = ORIGINAL_BIT_WIDTH * COMPRESSION_RATIO

    print(f"\n候选位宽: {TARGET_BIT_WIDTHS}")
    print(f"目标平均位宽: {target_avg_bit_width:.2f} bits")

    precision_loss_matrix = np.zeros((NUM_LAYERS, len(TARGET_BIT_WIDTHS)))
    for i in range(NUM_LAYERS):
        for j, bit in enumerate(TARGET_BIT_WIDTHS):
            precision_loss_matrix[i, j] = calculate_precision_loss(bit, ORIGINAL_BIT_WIDTH, sensitivities[i], ALPHA)
            
    qubo_model = kw.qubo.QuboModel()
    # 每层 2 个变量，无辅助变量
    y = kw.qubo.ndarray((NUM_LAYERS, 2), 'y', kw.qubo.Binary)

    all_terms = []
    
    # 因为没有辅助变量，不需要处理 scaling factor，直接使用大权重保证约束
    CONSTRAINT_PENALTY_WEIGHT = 50000.0

    for i in range(NUM_LAYERS):
        y0, y1 = y[i, 0], y[i, 1]
        loss_3, loss_4, loss_8 = precision_loss_matrix[i]
        
        # 1. 精度损失多项式 (严格对齐 3, 4, 8 映射)
        # l0 = loss_3
        # l1 = loss_4 - loss_3
        # l2 = loss_8 - loss_3
        # l12 = loss_3 - loss_4
        layer_precision_loss = (
            (loss_4 - loss_3) * y0 +
            (loss_8 - loss_3) * y1 +
            (loss_3 - loss_4) * y0 * y1 +
            loss_3 
        )
        all_terms.append(layer_precision_loss)
        
        # 2. 内存约束代数惩罚项
        # 基础位宽公式: b = 3 + y0 + 5*y1 - y0*y1
        wi = layer_weights[i]
        target = target_avg_bit_width
        b_const, b_y0, b_y1, b_y0y1 = 3, 1, 5, -1  # 核心系数替换
        
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

    original_qubo_mat = qubo_model.get_qubo_matrix()
    adjusted_qubo_mat = kw.qubo.adjust_qubo_matrix_precision(original_qubo_mat, bit_width=8)
    
    output_csv_path = f"{MODEL_NAME}_8bit_348opts.csv"
    pd.DataFrame(adjusted_qubo_mat).to_csv(output_csv_path, index=False, header=False)
    print(f"\n✅ 成功生成符合硬件要求的8-bit QUBO矩阵！文件已保存至: {os.path.abspath(output_csv_path)}")