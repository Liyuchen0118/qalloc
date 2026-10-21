import kaiwu as kw
import numpy as np
import os
import json
import pandas as pd

# ==============================================================================
# 模块一：数据加载与处理 (保持原样)
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
# 主程序：构建 8 选项完美降阶 QUBO
# ==============================================================================
if __name__ == "__main__":
    
    # ==========================================
    # 🛡️ 护城河：防止烧钱的测试开关
    # ==========================================
    TEST_MODE = False  # 首测请保持 True，确认解向量不坍缩后再改为 False
    
    MODEL_NAME = "qwen2.5-7b" 
    MODEL_INFO_PATH = f"model_info/{MODEL_NAME}"
    
    try:
        layer_params = load_and_flatten_json(os.path.join(MODEL_INFO_PATH, "LayerParams.json"))
        sensitivities = load_and_flatten_json(os.path.join(MODEL_INFO_PATH, "fisher.json"))
    except FileNotFoundError:
        NUM_LAYERS = 196
        layer_params = np.random.rand(NUM_LAYERS) * 1e7
        sensitivities = np.random.rand(NUM_LAYERS) * 0.1
        
    if TEST_MODE:
        print("\n[!!!] 开启了测试模式，仅提取前 10 层生成微型矩阵用于权重验证。")
        layer_params = layer_params[:10]
        sensitivities = sensitivities[:10]
        
    NUM_LAYERS = len(layer_params)

    # 目标参数
    TARGET_BIT_WIDTHS = [2, 3, 4, 5, 6, 7, 8, 16]
    ORIGINAL_BIT_WIDTH = 16
    ALPHA = 10.0
    COMPRESSION_RATIO = 0.25 # 目标平均 4-bit
    
    total_params = np.sum(layer_params, dtype=np.float64)
    layer_weights = layer_params / total_params
    target_avg_bit_width = ORIGINAL_BIT_WIDTH * COMPRESSION_RATIO

    print(f"\n候选位宽: {TARGET_BIT_WIDTHS}")
    print(f"参与计算层数: {NUM_LAYERS}")

    precision_loss_matrix = np.zeros((NUM_LAYERS, len(TARGET_BIT_WIDTHS)))
    for i in range(NUM_LAYERS):
        for j, bit in enumerate(TARGET_BIT_WIDTHS):
            precision_loss_matrix[i, j] = calculate_precision_loss(bit, ORIGINAL_BIT_WIDTH, sensitivities[i], ALPHA)
            
    qubo_model = kw.qubo.QuboModel()
    y = kw.qubo.ndarray((NUM_LAYERS, 4), 'y', kw.qubo.Binary)

    all_terms = []
    
    CONSTRAINT_PENALTY_WEIGHT = 2000.0
    AUX_PENALTY_WEIGHT = 800.0  # 沿用被验证成功的动态平衡权重

    for i in range(NUM_LAYERS):
        y0, y1, y2, z = y[i, 0], y[i, 1], y[i, 2], y[i, 3]
        L2, L3, L4, L5, L6, L7, L8, L16 = precision_loss_matrix[i]
        
        # 1. 精度损失多项式 (极度规整的完美映射，z = y0 * y1)
        l0 = L2
        l1 = L3 - L2
        l2 = L4 - L2
        l12 = L5 - L3 - L4 + L2
        l3 = L6 - L2
        l13 = L7 - L3 - L6 + L2
        l23 = L8 - L4 - L6 + L2
        l123 = L16 - L8 - L7 - L5 + L6 + L4 + L3 - L2
        
        layer_precision_loss = (
            l1 * y0 + l2 * y1 + l3 * y2 +
            l12 * z + l13 * y0 * y2 + l23 * y1 * y2 + l123 * z * y2 +
            l0
        )
        all_terms.append(layer_precision_loss)
        
        # 2. 内存约束二次展开 (完美位宽公式: b = 2 + y0 + 2y1 + 4y2 + 7*z*y2)
        wi = layer_weights[i]
        K = wi * (2 - target_avg_bit_width)
        C0, C1, C2 = wi, 2 * wi, 4 * wi
        Cz2 = 7 * wi
        
        const_coeff = K**2
        y0_coeff = C0**2 + 2*K*C0
        y1_coeff = C1**2 + 2*K*C1
        y2_coeff = C2**2 + 2*K*C2
        z_coeff = 2*C0*C1
        y0y2_coeff = 2*C0*C2
        y1y2_coeff = 2*C1*C2
        zy2_coeff = Cz2**2 + 2*K*Cz2 + 2*C0*Cz2 + 2*C1*Cz2 + 2*C2*Cz2
        
        layer_memory_penalty = (
            y0_coeff * y0 + y1_coeff * y1 + y2_coeff * y2 + z_coeff * z +
            y0y2_coeff * y0 * y2 + y1y2_coeff * y1 * y2 + zy2_coeff * z * y2 +
            const_coeff
        )
        all_terms.append(CONSTRAINT_PENALTY_WEIGHT * layer_memory_penalty)
        
        # 3. Rosenberg 惩罚项 (注意：这次约束的是 z = y0 * y1)
        aux_penalty = 3 * z + y0 * y1 - 2 * y0 * z - 2 * y1 * z
        all_terms.append(AUX_PENALTY_WEIGHT * aux_penalty)

    final_objective = kw.qubo.quicksum(all_terms)
    qubo_model.set_objective(final_objective)

    original_qubo_mat = qubo_model.get_qubo_matrix()
    adjusted_qubo_mat = kw.qubo.adjust_qubo_matrix_precision(original_qubo_mat, bit_width=8)
    
    file_suffix = "TEST_10layers" if TEST_MODE else "FULL"
    output_csv_path = f"qwen_algebraic_qubo_8bit_8opts_{file_suffix}.csv"
    pd.DataFrame(adjusted_qubo_mat).to_csv(output_csv_path, index=False, header=False)
    
    print(f"\n✅ 成功生成矩阵！文件已保存至: {os.path.abspath(output_csv_path)}")