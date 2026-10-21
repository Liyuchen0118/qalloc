import argparse
import json
import os
import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import copy
import numpy as np

# Configuration constants
PARAM_ORDER = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj"
]

def get_calib_dataset(data="wikitext2", tokenizer=None, n_samples=128, seq_len=2048):
    if data == "wikitext2":
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        encodings = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    else:
        raise ValueError(f"Dataset {data} not supported")

    input_ids = encodings.input_ids
    nsamples = n_samples
    
    batch = []
    for _ in range(nsamples):
        i = np.random.randint(0, input_ids.shape[1] - seq_len - 1)
        j = i + seq_len
        inp = input_ids[:, i:j]
        batch.append(inp)
    
    return torch.cat(batch, dim=0)

def get_layer_params(model):
    """
    Extracts parameter counts for each layer and targeted module.
    Structure: { "layer_idx": { "param_name": count } }
    """
    layer_params = {}
    
    # Iterate over layers
    # Assuming standard Llama structure: model.model.layers
    if hasattr(model, "model"):
        layers = model.model.layers
    else:
        # Fallback or for other architectures
        layers = model.layers

    for i, layer in enumerate(layers):
        layer_idx = str(i)
        layer_params[layer_idx] = {}
        
        # We manually map the module names to match the expected JSON keys
        # Llama modules: self_attn.q_proj, mlp.gate_proj, etc.
        
        modules = {
            "self_attn.q_proj": layer.self_attn.q_proj,
            "self_attn.k_proj": layer.self_attn.k_proj,
            "self_attn.v_proj": layer.self_attn.v_proj,
            "self_attn.o_proj": layer.self_attn.o_proj,
            "mlp.gate_proj": layer.mlp.gate_proj,
            "mlp.up_proj": layer.mlp.up_proj,
            "mlp.down_proj": layer.mlp.down_proj
        }
        
        for name, module in modules.items():
            layer_params[layer_idx][name] = module.weight.numel()
            
    return layer_params



def fake_quantize(tensor, bits=4):
    # Simple Min-Max Fake Quantization
    # x_q = round((x - min) / scale) * scale + min
    
    # Per-channel quantization is standard for LLMs (dim=0 or 1 usually out_features)
    # Llama Linear weights are (out_features, in_features)
    # Quantize along input dimension? Usually per-channel (row-wise)
    
    # Simplified simulation
    _min = tensor.min(dim=1, keepdim=True)[0]
    _max = tensor.max(dim=1, keepdim=True)[0]
    scale = (_max - _min) / (2**bits - 1)
    scale[scale == 0] = 1.0 # prevent div 0
    
    zero_point = -_min / scale
    
    q = torch.clamp(torch.round(tensor / scale + zero_point), 0, 2**bits - 1)
    dq = (q - zero_point) * scale
    return dq

def compute_perplexities(model, dataloader, device):
    """
    Computes loss (perplexity proxy) when each layer is individually quantized to 4-bit.
    Structure: { "4": { "layer_idx": { "param_name": value } } }
    """
    print("Computing Perplexities (Sensitivity)...")
    model.eval()
    
    # Calculate Baseline Loss
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            if len(batch.shape) == 1: batch = batch.unsqueeze(0)
            outputs = model(batch, labels=batch)
            total_loss += outputs.loss.item()
    baseline_loss = total_loss / len(dataloader)
    print(f"Baseline Loss: {baseline_loss:.4f}")
    
    if hasattr(model, "model"):
        layers = model.model.layers
    else:
        layers = model.layers
        
    perplexities = {"4": {}}
    
    # We use a subset for sensitivity to save time
    # (Optional: reduce dataloader size for this loop)
    # For extraction script, we keep it consistent.
    
    for i, layer in enumerate(layers):
        layer_idx = str(i)
        perplexities["4"][layer_idx] = {}
        
        modules = {
            "self_attn.q_proj": layer.self_attn.q_proj,
            "self_attn.k_proj": layer.self_attn.k_proj,
            "self_attn.v_proj": layer.self_attn.v_proj,
            "self_attn.o_proj": layer.self_attn.o_proj,
            "mlp.gate_proj": layer.mlp.gate_proj,
            "mlp.up_proj": layer.mlp.up_proj,
            "mlp.down_proj": layer.mlp.down_proj
        }
        
        for name, module in modules.items():
            print(f"Profiling Layer {i} - {name}...")
            
            # Backup weight
            orig_weight = module.weight.data.clone()
            
            # Fake Quantize
            module.weight.data = fake_quantize(module.weight.data, bits=4)
            
            # Measure Loss
            curr_loss = 0
            with torch.no_grad():
                for batch in dataloader:
                    batch = batch.to(device)
                    if len(batch.shape) == 1: batch = batch.unsqueeze(0)
                    outputs = model(batch, labels=batch)
                    curr_loss += outputs.loss.item()
            
            avg_loss = curr_loss / len(dataloader)
            perplexities["4"][layer_idx][name] = avg_loss
            
            # Restore weight
            module.weight.data = orig_weight
            
    return perplexities

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="", help="Path to the Llama model")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save JSON files")
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--n_samples", type=int, default=32, help="Number of samples for calibration")
    args = parser.parse_args()

    if not args.model_path:
        print("Error: --model_path is required. Please edit the script or provide the argument.")
        # For demonstration purposes in the generated file, we allow it to fail or mock
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from {args.model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Prepare Data
    print("Loading calibration dataset...")
    # NOTE: batch_size=1 is typical for PPL eval to avoid padding issues affecting loss calculation slightly
    # For Fisher, batching is fine.
    dataset = get_calib_dataset(tokenizer=tokenizer, n_samples=args.n_samples, seq_len=args.seq_len)
    # Create simple iterable
    dataloader = [dataset[i] for i in range(dataset.shape[0])]

    # 1. Layer Params
    print("Generating LayerParams.json...")
    layer_params = get_layer_params(model)
    with open(os.path.join(args.output_dir, "LayerParams.json"), "w") as f:
        json.dump(layer_params, f, indent=4)

    # 2. Fisher Info (Skipped)
    # print("Generating fisher.json...")
    # fisher_info = compute_fisher(model, dataloader, device)
    # with open(os.path.join(args.output_dir, "fisher.json"), "w") as f:
    #     json.dump(fisher_info, f, indent=4)

    # 3. Perplexities
    print("Generating perplexities.json...")
    ppl_info = compute_perplexities(model, dataloader, device)
    with open(os.path.join(args.output_dir, "perplexities.json"), "w") as f:
        json.dump(ppl_info, f, indent=4)

    print("Done! Files saved to", args.output_dir)

if __name__ == "__main__":
    main()
