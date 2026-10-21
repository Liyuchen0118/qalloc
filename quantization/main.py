import time
import torch
import torch.nn as nn
import argparse
import numpy as np
from loguru import logger

from mixq.layerwise_quant import *
from mixq.utils.misc import *
from mixq.utils.datautils import *
from mixq.utils.modelutils import *
from evaluation.perplexity import *
from evaluation.eval_wikitext import *
from lm_eval import evaluator

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # 模型路径：HuggingFace 模型或本地模型路径
    parser.add_argument(
        '--model', type=str, required=True,
        help='hugging face model to load or path to local model.'
    )

    # 校准数据集：用于提取校准数据的数据集，可选 wikitext2, ptb, c4, custom_path
    parser.add_argument(
        '--dataset', type=str,default='wikitext2',
        help='Where to extract calibration data from. choices = [wikitext2, ptb, c4, custom_path]'
    )

    # 数据集目录：本地数据集目录路径
    parser.add_argument(
        '--dataset_dir', type=str, default='',  # '/root/autodl-tmp/datasets/',
        help='load datasets from local directory.'
    )

    # 校准样本数：用于校准的样本数量
    parser.add_argument(
        '--nsamples', type=int, default = 8,
        help='Number of calibration data samples.'
    )

    # 层敏感度策略：确定各层量化策略的方法
    # - fisher: 使用 Fisher 信息估计层量化敏感度（推荐）
    # - activation: 量化前后激活值变化
    # - threshold: 阈值方法
    # - random: 随机分配
    # - ppl: 使用困惑度
    parser.add_argument(
        '--strategy', type=str, choices=['fisher', 'activation', 'threshold', 'random', "ppl"],
        help='The specific method for determining the layer quantization strategy, \
            "activation" indicates the change in activation value before and after quantization, \
            "fisher" indicates using fisher information to estimate the layer quantization sensitivity, \
            "random" indicates randomly assigning a quantization strategy.'
    )

    # 比特分配策略：基于 Fisher 的位宽分配优化算法
    # - greedy: 贪心算法
    # - genetic: 遗传算法
    # - rl: 强化学习
    # - annealing: 模拟退火
    # - bayesian: 贝叶斯优化
    # - random: 随机分配
    parser.add_argument(
        '--allocate_strategy', type=str , choices=['greedy', 'genetic', 'rl', 'annealing','bayesian','random'],
        help='Bit width allocation strategy based on Fisher'
    )

    # 是否加载已有的 Fisher 信息文件
    parser.add_argument(
        '--load_fisher', action='store_true',
        help='Whether to load a FisherInfo txt.'
    )

    # 是否加载已有的困惑度文件
    parser.add_argument(
        '--load_ppl', action='store_true',
        help='Whether to load a ppl txt.'
    )

    # 手动位宽分配：逗号分隔的比特位宽列表，对应每层的分配
    parser.add_argument(
        '--allocation', type=str, default=None,
        help='The bit allocation for each layer.'
    )

    # 可用比特位宽：可分配给各层的比特位宽候选集，如 3,4,5
    parser.add_argument(
        '--wbits', type=str, default="3,4,5",
        help='The number of bits to use for weight quantization; at least one lower bits.'
    )

    # 目标比特位宽：量化后模型的平均比特位宽目标
    parser.add_argument(
        '--target_bit', type=float, default=4,
        help='The target bit of total quantization.'
    )

    # 量化方法：权重量化算法
    # - gptq: GPTQ 量化
    # - awq: AWQ 量化
    # - owq: OWQ 量化
    # - nearest: 最近邻量化
    # - omni: OmniQuant 量化
    parser.add_argument(
        '--quant_method', type=str, default='', choices=['gptq', 'awq', 'owq', 'nearest','omni'],
        help='Choosing an appropriate quantification method.\
            Different methods have different processes in handling quantization parameters.'
    )

    # Alpha 参数：目标函数中的权衡参数，用于控制压缩率与精度的平衡
    parser.add_argument(
        '--alpha', type=float, default=0.75,
        help='Hyperparameters used to calculate the objective function'
    )

    # Top R 比例：使用 ppl 策略时考虑的顶层敏感层比例
    parser.add_argument(
        '--top_r', type=float, default=0.1,
        help='The ratio of the top layers to be considered for ppl allocation.'
    )

    # 测试比特位宽：计算 Fisher 信息时使用的比特位宽
    parser.add_argument(
        '--test_bit', type=str, default="4",
        help='The bits to calculate fisher info.'
    )

    # 扰动幅度：量化扰动相对于原始权重的幅度比例
    parser.add_argument(
        '--perturb_percentage', type=float, default=0.1,
        help='The amplitude of the added perturbation is -1 to 1 times the original model weight.'
    )

    # 使用原始模型：不进行量化，直接使用原始模型
    parser.add_argument(
        '--original', action='store_true',
        help='whether to use original model.'
    )

    # GPU 设备编号：指定使用的 GPU 设备
    parser.add_argument(
        '--device', type=int, default=None,
        help='GPUs to use'
    )

    # 量化参数调优方法
    # - mse: 均方误差（推荐）
    # - minmax: 最小最大值
    parser.add_argument(
        '--tuning', type=str, default='mse', choices=['mse', 'minmax'],
        help='Method for quantization parameter tuning.'
    )

    # 模型数据类型：如 bfloat16，用于特定模型（如 falcon、llama65b）
    parser.add_argument(
        '--dtype', type=str, default=None,
        help='Data type of model. Use bfloat16 for falcon model family or llama 65B model'
    )

    # 指定层：只对特定层应用量化
    parser.add_argument(
        '--layers', nargs='+', type=str, default=None,
        help='Layers to apply MixQuant.'
    )

    # 同名层重置：是否在同一名称的层之间重置比特位
    parser.add_argument(
        '--sameLayerReset', action='store_true',
        help='Whether to reset bits between same name layer.'
    )

    # 随机种子：用于校准数据采样和随机层策略
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Seed for sampling the calibration data and the random layerwise quantization strategy.'
    )

    # 对称量化：是否使用对称量化（而非非对称）
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )

    # 量化组大小：细粒度量化的组大小，默认 128
    parser.add_argument(
        '--groupsize', type=int, default=128,
        help='Groupsize for fine-grained quantization; default uses full row.'
    )

    # 保存路径：量化后模型的保存路径
    parser.add_argument(
        '--save_path', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )

    # 假量化：是否保存假量化模型（仅模拟量化效果）
    parser.add_argument(
        '--fake', action='store_true',
        help='Whether to save fake quantized model.'
    )

    # 加载路径：加载已量化模型的路径
    parser.add_argument(
        '--load', type=str, default='',
        help='Load fake quantized checkpoint.'
    )

    # 日志文件：日志输出文件名
    parser.add_argument(
        '--logfile', type=str, default='',
        help='Logging file name'
    )

    # 评估困惑度：量化后评估 wikitext2、ptb、c4 上的困惑度
    parser.add_argument(
        '--evaluate_ppl', action='store_true',
        help='Whether to evaluation model, adding parameters not to test.'
    )

    # 评估推理任务：使用推理任务（boolq、arc、hellaswag等）评估模型
    parser.add_argument(
        '--reasoning', action='store_true',
        help='Whether to evaluation model using reasoning tasks.'
    )

    # 信任远程代码：是否信任远程代码
    parser.add_argument(
        '--trust_remote_code', action='store_true',
    )

    # 真实顺序：是否按真实顺序处理层
    parser.add_argument(
        '--true-sequential', action='store_true',
        help='Whether to run in true sequential model.'
    )

    # 加载量化模型：是否加载已量化的模型
    parser.add_argument(
        '--quant', action='store_true',
        help='Whether to load a quantilized model.'
    )

    # AWQ 版本：AWQ 量化网络架构版本
    parser.add_argument(
        '--version', type=str, default='gemm',
        help='Awq quantized network architecture.'
    )

    args = parser.parse_args()
    meta = processing_arguments(args)
    args.meta = meta
    seed_all(args.seed)

    logger.info('loading model ...')
    if args.load:
        model = load_model(args.model, args.load, args.faster)
    else:
        model = get_hfmodel(args.model, args.dtype, args.quant, trust_remote_code=args.trust_remote_code)
    logger.info('Model loaded.')

    if getattr(model.config, 'max_position_embeddings', None):
        args.seqlen = model.config.max_position_embeddings
    elif getattr(model.config, 'max_sequence_length', None):
        args.seqlen = model.config.max_sequence_length
    else:
        args.seqlen = 2048

    args.seqlen = 2048

    # implementation of quantization
    if not args.load and args.wbits and not args.original:
        logger.info(f'The model needs quantilization, start loading {args.dataset} as validation data ...')
        dataloader = get_loaders(
            args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, train=True, local_dir=args.dataset_dir
        )
        logger.info('Validation Data loaded.')

        t_start = time.time()
        quantizers = layerwise_quantize(model, dataloader, args)
        if args.save_path:
            save_model(model, args.save_path, args.model, args.fake)

        t = round((time.time() - t_start),1)
        logger.info(f"Running Time : {t} s")

    torch.cuda.empty_cache()

    # evaluation perplexity
    if args.evaluate_ppl:
        ppl_tasks = ['wikitext2','ptb','c4']
        for dataset in ppl_tasks:
            testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=args.seqlen, train=False, local_dir=args.dataset_dir
            )
            t1 = time.time()
            logger.info(f'Evaluating {dataset} sequence {args.seqlen} model_path {args.model}')
            ppl_score = eval_ppl(model, testloader, args.device, args)
            t2 = time.time() - t1
            logger.info(f'{dataset} perplexity: {ppl_score}')
            logger.info(f"Evaluation time: {t2:.2f} seconds")
        model.to('cpu')

    # evaluation reasoning
    if args.reasoning:
        results = {}
        from evaluation.lm_model import HFModelWrapper
        reason_tasks = ["boolq","arc_easy","arc_challenge","hellaswag","winogrande"]
        # reason_tasks = ["arc_easy","arc_challenge"]#,
        # reason_tasks = ["hellaswag","winogrande"]
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
        lm= HFModelWrapper(model, tokenizer)
        t_results = evaluator.simple_evaluate(
            lm,
            tasks = reason_tasks,
            num_fewshot=0,
            limit=None,
        )
        results.update(t_results["results"])
        logger.info(results)
        try:
            acc_values = [task_result["acc,none"] for task_result in results.values() if "acc,none" in task_result]
            avg_acc = sum(acc_values) / len(acc_values)
            logger.info(f"avg acc: {avg_acc:.4f}")
        except Exception as e:
            logger.error(f"Error calculating average accuracy: {e}")
        del lm
        # from evaluation.reason import *
        # reason_tasks = ['BoolQ','ARC-E','ARC-C','HellaSwag','WinoGrande']
        # tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
        # scores = reason_test(model, tokenizer)
        # # 记录平均值
        # logger.info(f"average score: {np.mean(scores)}")
