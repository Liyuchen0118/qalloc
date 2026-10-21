import optuna
from optuna.trial import TrialState
import numpy as np
import json
import warnings
from functools import partial
import time
import os
from typing import Tuple, List, Dict

# ==============================================================================
# 模块一：数据加载与处理工具 (与QUBO脚本对齐)
# ==============================================================================
param_order = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"
]

def load_and_flatten_json(file_path: str) -> np.ndarray:
    """
    从指定路径加载JSON文件并将其值平铺成一维Numpy数组。
    此版本已更新，可以处理类似fisher.json中的额外嵌套层。
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
# 模块二：百分制评分指标 (与QUBO脚本对齐)
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
    w_acc: float = 0.7, # 默认更看重精度
    w_mem: float = 0.3
) -> Dict[str, float]:
    """
    计算一个给定分配方案的百分制综合评分。
    """
    num_layers = len(layer_params)

    # 1. 计算基准值
    loss_max = np.sum([calculate_precision_loss(min(target_bit_widths), original_bit_width, sensitivities[i], alpha) for i in range(num_layers)])
    loss_min = np.sum([calculate_precision_loss(max(target_bit_widths), original_bit_width, sensitivities[i], alpha) for i in range(num_layers)])
    
    # 2. 计算当前方案的精度损失
    current_loss = np.sum([calculate_precision_loss(final_assigned_bits[i], original_bit_width, sensitivities[i], alpha) for i in range(num_layers)])

    # 3. 计算精度得分
    score_acc = 100.0 * (1 - (current_loss - loss_min) / (loss_max - loss_min)) if (loss_max - loss_min) > 1e-6 else 100.0

    # 4. 计算当前方案的内存偏差
    actual_avg_bit_width = np.average(final_assigned_bits, weights=layer_params)
    delta_b = abs(actual_avg_bit_width - target_avg_bit_width)
    
    # 5. 计算内存收敛得分
    delta_b_max = max(abs(max(target_bit_widths) - target_avg_bit_width), abs(min(target_bit_widths) - target_avg_bit_width))
    score_mem = 100 * max(0, 1 - (delta_b / delta_b_max)) if delta_b_max > 1e-6 else 100.0

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
# 模块三：贝叶斯优化器
# ==============================================================================
class BayesianOptimization:
    def __init__(self, bits, p, F, B, R, alpha, n_trials=8000):
        self.bits = bits
        self.p = p
        self.F = F
        self.B = B
        self.R = R
        self.alpha = alpha
        self.n_trials = n_trials
        
        self.P_total = np.sum(p, dtype=np.float64) 
        self.target_size = self.P_total * self.B * self.R
        
        self.best_allocation = None
        self.study = optuna.create_study(directions=["minimize", "minimize"])

    def objective(self, trial):
        bit_allocation = [trial.suggest_categorical(f"bit_{i}", self.bits) for i in range(len(self.F))]
        bit_allocation = np.array(bit_allocation)
        
        objective_value = sum(
            calculate_precision_loss(bit, self.B, sensitivity, self.alpha)
            for sensitivity, bit in zip(self.F, bit_allocation)
        )
        
        compressed_size = np.sum(self.p * bit_allocation)
        constraint_violation = np.abs(compressed_size - self.target_size)
        
        return objective_value, constraint_violation

    def run(self):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)
        
        all_trials = self.study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
        feasible_trials = [t for t in all_trials if t.values[1] <= 1e-3]
        
        if feasible_trials:
            print("\n✅ 成功找到满足约束的解！")
            best_trial_obj = min(feasible_trials, key=lambda t: t.values[0])
        else:
            warnings.warn("⚠️ 未找到严格满足约束的解，将返回最接近目标的解。")
            best_trial_obj = min(all_trials, key=lambda t: t.values[1])
            
        self.best_allocation = np.array([best_trial_obj.params[f"bit_{i}"] for i in range(len(self.F))])

    def get_best_allocation(self):
        return self.best_allocation

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
    N_TRIALS = 7000

    # --- 步骤2: 加载数据 ---
    print(f"--- 正在从 '{MODEL_INFO_PATH}' 加载数据 ---")
    p = load_and_flatten_json(os.path.join(MODEL_INFO_PATH, "LayerParams.json"))
    F = load_and_flatten_json(os.path.join(MODEL_INFO_PATH, "fisher.json"))
    
    assert len(p) == len(F), "数据长度不匹配！"
    print(f"成功加载了 {len(p)} 个层的指标。")

    # --- 步骤3: 运行贝叶斯优化 ---
    print(f"\n--- 开始运行贝叶斯优化 ({N_TRIALS}轮) ---")
    t_start = time.time()
    bayesian_opt = BayesianOptimization(bits, p, F, B, R, alpha, n_trials=N_TRIALS)
    bayesian_opt.run()
    t_end = time.time()
    
    best_allocation = bayesian_opt.get_best_allocation()

    # --- 步骤4: 输出结果与评估 ---
    if best_allocation is not None:
        print("\n" + "="*50)
        print("           贝叶斯优化最终结果")
        print("="*50)
        
        target_avg_bit_width = B * R
        scores = calculate_final_score(
            final_assigned_bits=best_allocation,
            sensitivities=F,
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

        # ==================== 新增：输出完整列表 ====================
        print("\n--- 完整位宽分配列表 ---")
        # 使用 .tolist() 转换numpy数组为纯Python列表，以便于复制
        print(best_allocation.tolist())
        # ==========================================================

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
