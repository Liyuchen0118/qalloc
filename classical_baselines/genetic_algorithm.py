import random
import numpy as np
from deap import base, creator, tools, algorithms
import json
import os
import time
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
    w_acc: float = 0.7,
    w_mem: float = 0.3
) -> Dict[str, float]:
    """
    计算一个给定分配方案的百分制综合评分。
    """
    num_layers = len(layer_params)
    loss_max = np.sum([calculate_precision_loss(min(target_bit_widths), original_bit_width, sensitivities[i], alpha) for i in range(num_layers)])
    loss_min = np.sum([calculate_precision_loss(max(target_bit_widths), original_bit_width, sensitivities[i], alpha) for i in range(num_layers)])
    current_loss = np.sum([calculate_precision_loss(final_assigned_bits[i], original_bit_width, sensitivities[i], alpha) for i in range(num_layers)])
    score_acc = 100.0 * (1 - (current_loss - loss_min) / (loss_max - loss_min)) if (loss_max - loss_min) > 1e-6 else 100.0
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
# 模块三：遗传算法求解器
# ==============================================================================
class GeneticAlgorithm:
    def __init__(self, bits, p, F, B, R, alpha, 
                 population_size=100, max_generations=50, cxpb=0.7, mutpb=0.2, elite_size=10):
        self.bits = bits
        self.p = p
        self.F = F
        self.B = B
        self.R = R
        self.alpha = alpha
        self.population_size = population_size
        self.max_generations = max_generations
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.elite_size = elite_size
        self.N = len(p)
        self.best_individual = None
        
        self.target_avg_bit_width = self.B * self.R

        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_bit", random.choice, self.bits)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_bit, n=self.N)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.evaluate_individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        # ==================== 核心修复：使用自定义变异函数 ====================
        self.toolbox.register("mutate", self.custom_mutate, indpb=0.1)
        # ====================================================================
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def custom_mutate(self, individual, indpb):
        """自定义变异操作，确保新值仍在候选位宽列表中"""
        for i in range(len(individual)):
            if random.random() < indpb:
                # 从 bits 列表中随机选择一个不同于当前值的位宽
                possible_choices = [b for b in self.bits if b != individual[i]]
                if possible_choices:
                    individual[i] = random.choice(possible_choices)
        return individual,

    def evaluate_individual(self, individual):
        """评估个体的适应度，包含对约束的惩罚"""
        objective = sum(
            calculate_precision_loss(bit, self.B, sensitivity, self.alpha)
            for sensitivity, bit in zip(self.F, individual)
        )
        current_avg_bit_width = np.average(individual, weights=self.p)
        constraint_violation = (current_avg_bit_width - self.target_avg_bit_width)**2
        penalty_weight = 10000.0 
        return objective + penalty_weight * constraint_violation,

    def run(self):
        """运行遗传算法"""
        pop = self.toolbox.population(n=self.population_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        
        pop, log = algorithms.eaSimple(pop, self.toolbox, cxpb=self.cxpb, mutpb=self.mutpb, 
                                             ngen=self.max_generations, stats=stats, halloffame=hof, verbose=True)
        
        self.best_individual = hof[0]
        return self.best_individual

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
    
    POPULATION_SIZE = 200
    MAX_GENERATIONS = 100

    # --- 步骤2: 加载数据 ---
    print(f"--- 正在从 '{MODEL_INFO_PATH}' 加载数据 ---")
    p = load_and_flatten_json(os.path.join(MODEL_INFO_PATH, "LayerParams.json"))
    F_raw = load_and_flatten_json(os.path.join(MODEL_INFO_PATH, "fisher.json"))
    
    assert len(p) == len(F_raw), "数据长度不匹配！"
    print(f"成功加载了 {len(p)} 个层的指标。")

    # --- 步骤3: 运行遗传算法 ---
    print(f"\n--- 开始运行遗传算法 (种群: {POPULATION_SIZE}, 代数: {MAX_GENERATIONS}) ---")
    t_start = time.time()
    ga = GeneticAlgorithm(bits, p, F_raw, B, R, alpha, 
                          population_size=POPULATION_SIZE, max_generations=MAX_GENERATIONS)
    best_allocation = ga.run()
    t_end = time.time()
    
    # --- 步骤4: 输出结果与评估 ---
    if best_allocation is not None:
        print("\n" + "="*50)
        print("           遗传算法最终结果")
        print("="*50)
        
        target_avg_bit_width = B * R
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
        print("\n❌ 优化过程未能找到任何有效解。")
