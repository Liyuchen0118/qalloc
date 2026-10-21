"""
评估已量化模型的 PPL 脚本
用法: python eval_quantized.py --model_path ./result
"""
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from mixq.utils.datautils import get_loaders
from evaluation.perplexity import eval_ppl
from loguru import logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./result',
                        help='量化模型路径')
    parser.add_argument('--dataset', type=str, default='wikitext2',
                        help='评估数据集')
    parser.add_argument('--nsamples', type=int, default=None,
                        help='评估样本数，默认全部')
    parser.add_argument('--seqlen', type=int, default=2048,
                        help='序列长度')
    args = parser.parse_args()

    # 加载量化模型
    logger.info(f"Loading quantized model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True
    )
    logger.info("Model loaded.")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # 加载数据集
    logger.info(f"Loading dataset: {args.dataset}...")
    testloader = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=0,
        model=args.model_path,
        seqlen=args.seqlen,
        train=False
    )
    logger.info("Dataset loaded.")

    # 评估 PPL
    logger.info(f"Evaluating {args.dataset}...")
    ppl_score = eval_ppl(model, testloader, device=0, args=args)
    logger.info(f"{args.dataset} perplexity: {ppl_score}")

    # 评估多个数据集
    datasets = ['wikitext2', 'ptb', 'c4']
    results = {}

    for dataset in datasets:
        try:
            logger.info(f"Evaluating {dataset}...")
            testloader = get_loaders(
                dataset,
                nsamples=args.nsamples,
                seed=0,
                model=args.model_path,
                seqlen=args.seqlen,
                train=False
            )
            ppl_score = eval_ppl(model, testloader, device=0, args=args)
            results[dataset] = ppl_score
            logger.info(f"{dataset} perplexity: {ppl_score}")
        except Exception as e:
            logger.error(f"Error evaluating {dataset}: {e}")

    # 打印汇总结果
    logger.info("\n" + "="*50)
    logger.info("Evaluation Results Summary:")
    logger.info("="*50)
    for dataset, ppl in results.items():
        logger.info(f"{dataset}: {ppl}")
    logger.info("="*50)

if __name__ == '__main__':
    main()
