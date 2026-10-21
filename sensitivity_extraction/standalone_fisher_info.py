import argparse
import copy
import gc
import json
import math
import os
import random
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIGURATION (Mocking model_config.json) ---
MODEL_META = {
    "llama": {
        "map_layer": {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj", "o": "self_attn.o_proj", "up": "mlp.up_proj", "gate": "mlp.gate_proj", "down": "mlp.down_proj"},
        "sequential": [
            ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
            ["self_attn.o_proj"],
            ["mlp.up_proj", "mlp.gate_proj"],
            ["mlp.down_proj"]
        ],
        "ln_layers": ["input_layernorm", "post_attention_layernorm"],
        "layers": "model.model.layers",
        "prefix": "model.layers",
        "pre_layers": ["model.model.embed_tokens"],
        "post_layers": ["model.model.norm"],
        "inp_kwargs": ["attention_mask", "position_ids"]
    },
    "qwen": {
        "map_layer": {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj", "o": "self_attn.o_proj", "up": "mlp.up_proj", "gate": "mlp.gate_proj", "down": "mlp.down_proj"},
        "sequential": [
            ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
            ["self_attn.o_proj"],
            ["mlp.up_proj", "mlp.gate_proj"],
            ["mlp.down_proj"]
        ],
        "ln_layers": ["input_layernorm", "post_attention_layernorm"],
        "layers": "model.model.layers",
        "prefix": "model.layers",
        "pre_layers": ["model.model.embed_tokens"],
        "post_layers": ["model.model.norm"],
        "inp_kwargs": ["attention_mask", "position_ids"]
    },
}

# --- UTILS ---

def find_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def parsing_layers(model, meta):
    results = OrderedDict({'layers': None, 'pre_layers': [], 'post_layers': []})
    for data_name in results.keys():
        data = meta[data_name]
        if isinstance(data, list):
            for data_ in data:
                root_attr = model
                attrs = data_.split('.')[1:]
                for attr in attrs:
                    root_attr = getattr(root_attr, attr)
                results[data_name].append(root_attr)
        else:  # str
            root_attr = model
            attrs = data.split('.')[1:]
            for attr in attrs:
                root_attr = getattr(root_attr, attr)
            results[data_name] = root_attr

    return results.values()

def create_position_ids(length: int, device: str = "cpu") -> torch.Tensor:
    position_ids = torch.arange(0, length, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0)
    return position_ids

def split_model_across_gpus(model, meta, num_gpus):
    layers, pre_layers, post_layers = parsing_layers(model, meta)
    
    # Pre-layers on GPU 0
    for i, pre_layer in enumerate(pre_layers):
        pre_layers[i] = pre_layer.to('cuda:0')
    
    # Convert layers to list explicitly
    layers = list(layers)
    
    num_transformer_layers = len(layers)
    if num_gpus > 0:
        layers_per_gpu = math.ceil(num_transformer_layers / num_gpus)
    else:
        layers_per_gpu = num_transformer_layers

    # Start with pre_layers wrapped in a ModuleList or similar if they are multiple
    # The original code uses `model_parts = pre_layers` where pre_layers is a list.
    # Then it appends transformer blocks (as lists or Sequentials?).
    # Original: `model_part = nn.Sequential(*layers[...])` if it works, BUT
    # Original code loop: `for layer in model_part: outputs = layer(...)`
    # nn.Sequential IS iterable. So `for layer in sequential` works.
    
    model_parts = []
    # Add pre-layers as a single Sequential block
    if pre_layers:
        model_parts.append(nn.Sequential(*pre_layers))

    for i in range(num_gpus):
        start_idx = i * layers_per_gpu
        end_idx = min((i + 1) * layers_per_gpu, num_transformer_layers)
        if start_idx >= num_transformer_layers: 
             break
        
        device = f'cuda:{i}'
        block_layers = layers[start_idx:end_idx]
        for layer in block_layers:
            layer.to(device)
        
        # We assume transformer layers can handle standard forward or we iterate manually
        # To support **kwargs loop in evaluate func, we need to return something iterable.
        # nn.Sequential is iterable.
        model_part = nn.Sequential(*block_layers) 
        model_parts.append(model_part)

    # Post-layers
    last_gpu = f'cuda:{num_gpus-1}' if num_gpus > 0 else 'cpu'
    for i, post_layer in enumerate(post_layers):
        post_layers[i] = post_layer.to(last_gpu)
    
    if post_layers:
        model_parts.append(nn.Sequential(*post_layers))

    return model_parts

def save_data(data, filename):
    def make_serializable(item):
        if isinstance(item, dict):
            return {key: make_serializable(value) for key, value in item.items()}
        elif isinstance(item, list):
            return [make_serializable(element) for element in item]
        elif hasattr(item, 'item'):
            return item.item()
        else:
            return item

    try:
        serializable_data = make_serializable(data)
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(serializable_data, f, indent=4)
        logger.info(f"Data saved in {filename}.")
    except Exception as e:
        logger.error(f"Failed to save data to {filename}: {e}")

def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


# --- QUANTIZATION CLASSES ---

def quantize_val(x, scale, zero, minq, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, minq, maxq)
    return scale * (q - zero)

def quantize_efficient(x_round, scale, zero, minq, maxq):
    q = torch.clamp(x_round + zero, minq, maxq)
    return scale * (q - zero)

class Quantizer(nn.Module):
    def __init__(self, bits, perchannel=False, sym=False, mse=False, norm=2.4):
        super(Quantizer, self).__init__()
        self.register_buffer('scale', torch.zeros(1))
        self.register_buffer('zero', torch.zeros(1))
        self.bits = bits
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.perchannel = perchannel
        self.set_bits(bits)
        self.num = 100
        self.eps = torch.tensor(1e-8)
    
    def set_bits(self, bits):
        self.bits = bits
        self.n_levels = 2 ** bits
        if self.sym:
            self.minq, self.maxq = -((self.n_levels - 1) // 2 + 1), (self.n_levels - 1) // 2
        else:
            self.minq, self.maxq = 0, self.n_levels - 1

    def lp_loss(self, pred, tgt, p=2.0):
        x = (pred - tgt).abs().pow(p)
        if not self.perchannel:
            return x.mean()
        else:
            y = torch.flatten(x, 1)
            return y.mean(1)
    
    def find_params(self, x, weight=False, num=100):
        self.num = num
        dev = x.device
        minq, maxq = self.minq, self.maxq
        
        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3]).flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)
        
        if self.mse:
             if self.perchannel:
                new_shape = [-1] + [1] * (len(x.shape) -  1)
            
             best_score = torch.zeros_like(xmin) + (1e+10)
             best_min = xmin.clone()
             best_max = xmax.clone()
            
             if self.sym:
                xrange = torch.max(xmin.abs(), xmax)
                zero = torch.zeros_like(xmin)
                if self.perchannel:
                    zero = zero.reshape(new_shape)
                for i in range(1, self.num + 1):
                    tmp_max = xrange / self.num * i
                    scale = torch.max(tmp_max / -minq, self.eps)
                    if self.perchannel:
                        scale = scale.reshape(new_shape)
                    x_round = torch.round(x / scale)
                    x_q = quantize_efficient(x_round, scale, zero, minq, maxq)
                    score = self.lp_loss(x, x_q, 2.4)
                    best_max = torch.where(score < best_score, tmp_max, best_max)
                    best_score = torch.min(score, best_score)
                
                max_val = torch.max(best_max, torch.zeros_like(best_max))

                self.scale = torch.max(max_val / -minq, self.eps)
                self.zero = torch.zeros_like(self.scale)
             else:
                xrange = xmax - xmin
                tmp_min = torch.zeros_like(xmin)
                for i in range(1, self.num + 1):
                    tmp_max = xrange / self.num * i
                    scale = torch.max((tmp_max - tmp_min) / (maxq - minq), self.eps)
                    delta = scale.clone()
                    if self.perchannel:
                        scale = scale.reshape(new_shape)
                    x_round = torch.round(x / scale)
                    for zp in range(0, self.n_levels):
                        new_min = tmp_min - zp * delta
                        new_max = tmp_max - zp * delta
                        zero = torch.clamp(minq - torch.round(new_min / delta), minq, maxq)
                        if self.perchannel:
                            zero = zero.reshape(new_shape)
                        x_q = quantize_efficient(x_round, scale, zero, minq, maxq)
                        score = self.lp_loss(x, x_q, 2.4)
                        best_min = torch.where(score < best_score, new_min, best_min)
                        best_max = torch.where(score < best_score, new_max, best_max)
                        best_score = torch.min(best_score, score)
            
                min_val_neg = torch.min(best_min, torch.zeros_like(best_min))
                max_val_pos = torch.max(best_max, torch.zeros_like(best_max))

                self.scale = torch.max((max_val_pos - min_val_neg) / (maxq - minq), self.eps)
                self.zero = torch.clamp(minq - torch.round(min_val_neg / self.scale), minq, maxq)
        else:
            if self.sym:
                xmax = torch.maximum(torch.abs(xmin), xmax)
                tmp = xmin < 0
                if torch.any(tmp):
                    xmin[tmp] = -xmax[tmp]
            tmp = (xmin == 0) & (xmax == 0) 
            xmin[tmp] = -1
            xmax[tmp] = +1
            if self.sym:
                self.scale = xmax / -minq
                self.zero = torch.zeros_like(self.scale)
            else:
                self.scale = (xmax - xmin) / maxq
                self.zero = torch.round(-xmin / self.scale)
        
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1)) 
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return quantize_val(x, self.scale, self.zero, self.minq, self.maxq)
        return x

    def ready(self):
        return torch.all(self.scale != 0)

class Mixq_Linear:
    def __init__(self, layer, wbits):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d): 
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        
        self.rows = W.shape[0]
        self.columns = W.shape[1]  
        self.nsamples = 0 
        self.bits = wbits
        self.quantizer = None
        self.quant_loss_threshold = None

    def fasterquant(self, quantbit, blocksize=128, groupsize=-1):
        self.quantizer.set_bits(quantbit)
        if self.quantizer:
            self.quantizer.find_params(self.layer.weight.data, weight=True, num=16)

        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()
        
        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)
        
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)

            for i in range(count):
                w = W1[:, i]
                if groupsize != -1:
                    if (i1 + i) % groupsize == 0:
                        self.quantizer.find_params(W[:, (i1 + i):min((i1 + i + groupsize),(self.columns))], weight=True, num=8)
                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 

        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        return Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype), torch.sum(Losses).item()

    def free(self):
        torch.cuda.empty_cache()

def Quantization_perturbation(bit, layer):
    mixq_linear = Mixq_Linear(layer, bit)
    mixq_linear.quantizer = Quantizer(bit, perchannel=True)
    original_weight = layer.weight.data.clone().to(torch.float)
    W_quant, _ = mixq_linear.fasterquant(quantbit=bit, groupsize=128)
    perturbation = W_quant - original_weight
    del original_weight, W_quant
    torch.cuda.empty_cache()
    return perturbation

# --- CORE FISHER LOGIC ---

def compute_manual_rope(seq_len, head_dim, device):
    theta = 10000.0
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float().to(device) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    # Return (1, seq_len, head_dim) - LlamaAttention will unsqueeze to (1, 1, seq_len, head_dim)
    return emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)

def evaluate_fisher_information_with_quant_perturb_sub_block(model, dataset, args):
    num_gpus = torch.cuda.device_count()
    meta = args.meta
    
    model_parts = split_model_across_gpus(model, meta, num_gpus)
    
    # Get head_dim from config
    head_dim = 128 # Default LLaMA
    if hasattr(model.config, "hidden_size") and hasattr(model.config, "num_attention_heads"):
        head_dim = model.config.hidden_size // model.config.num_attention_heads
    
    layer_batch = 1
    est_fisher_info = {}
    original_fisher = {}
    modified_fisher = {}
    original_perplexitys = []
    modified_perplexitys = {}
    
    # Assuming batch size is consistent
    batch_num = dataset[0].shape[0] if isinstance(dataset[0], torch.Tensor) else dataset[0]['input_ids'].shape[0]

    epoch_iterator = tqdm(dataset, desc="Iteration")
    for model_part in model_parts:
        model_part.train()

    # 1. Calculate Original Fisher and Perplexity
    for step, batch in enumerate(epoch_iterator):
        if isinstance(batch, dict):
            input_ids = batch['input_ids'].to('cuda:0')
        else:
            input_ids = batch[0].to('cuda:0') if isinstance(batch, (tuple, list)) else batch.to('cuda:0')
        
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
            
        batch_num = input_ids.size(0)

        for i in range(batch_num):
            inputs = input_ids[i].unsqueeze(0)
            inp_kwargs = {
                "attention_mask": None,
                "position_ids": create_position_ids(len(input_ids[i]), device=args.device)
            }
            
            # Forward
            outputs = inputs
            
            for j, model_part in enumerate(model_parts):
                if j == 0:  # Pre-layers
                    device = 'cuda:0'
                    outputs = model_part(outputs.to(device))
                    
                    # Compute Rotary Embeddings MANUALLY
                    if "position_embeddings" not in inp_kwargs:
                        try:
                            # Manually compute cos/sin with shape (1, seq_len, head_dim)
                            # This avoids LlamaRotaryEmbedding signature issues and shape mismatches
                            seq_len = outputs.shape[1]
                            cos, sin = compute_manual_rope(seq_len, head_dim, outputs.device)
                            # Cast to output dtype
                            cos = cos.to(dtype=outputs.dtype)
                            sin = sin.to(dtype=outputs.dtype)
                            inp_kwargs["position_embeddings"] = (cos, sin)
                        except Exception as e:
                            logger.warning(f"Could not compute manual rotary embeddings: {e}")
                            
                elif j == len(model_parts) - 1:  # Post-layers
                    device = f'cuda:{num_gpus - 1}' if num_gpus > 0 else 'cpu'
                    outputs = model_part(outputs.to(device))
                else:  # Transformer layers
                    device = f'cuda:{j % num_gpus}' if num_gpus > 0 else 'cpu'
                    
                    # Move kwargs to device
                    for item in inp_kwargs:
                        val = inp_kwargs[item]
                        if val is not None:
                            if isinstance(val, tuple): # Handle (cos, sin)
                                inp_kwargs[item] = tuple(v.to(device) for v in val)
                            elif isinstance(val, torch.Tensor):
                                inp_kwargs[item] = val.to(device)
                                
                    # Iterate layers in Sequential
                    for layer in model_part:
                        outputs = layer(outputs.to(device), **inp_kwargs)[0]

            model.lm_head.to(f'cuda:{num_gpus - 1}' if num_gpus > 0 else 'cpu')
            logits = model.lm_head(outputs)
            inputs = inputs.to(f'cuda:{num_gpus - 1}' if num_gpus > 0 else 'cpu')
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = inputs[:, 1:]

            loss_fn = nn.CrossEntropyLoss()
            perplexity = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            original_perplexitys.append(perplexity.item()) # Store item to save memory

            # Backward for Original Fisher
            for model_part in model_parts:
                model_part.zero_grad()
            perplexity.backward()

            # Accumulate Fisher
            block_number = 0
            for k, model_part in enumerate(model_parts):
                # Skip pre/post layers for Fisher (usually only optimize Transformer blocks)
                if k == 0 or k == len(model_parts) - 1:
                    continue
                # Sequential iterable
                for seq_layer in model_part:
                    if block_number not in original_fisher:
                        original_fisher[block_number] = {}
                    target_layers = find_layers(seq_layer)
                    
                    for name, param in seq_layer.named_parameters():
                        # name is like "self_attn.q_proj.weight"
                        if '.' in name:
                            layer_name, suffix = name.rsplit('.', 1)
                        else:
                            layer_name, suffix = name, ""
                            
                        if layer_name not in target_layers.keys() or 'bias' in suffix:
                            continue
                        if param.requires_grad and param.grad is not None:
                            grad_sq = torch.sum(param.grad.detach() ** 2).to("cpu")
                            if layer_name not in original_fisher[block_number]:
                                original_fisher[block_number][layer_name] = grad_sq
                            else:
                                original_fisher[block_number][layer_name] += grad_sq
                    block_number += 1

            # Cleanup
            for model_part in model_parts:
                model_part.zero_grad()
            del outputs, logits, shift_logits, shift_labels, perplexity
        torch.cuda.empty_cache()

    # Normalize Original Fisher
    total_samples = len(dataset) * batch_num # Approx if batch_num varies
    for block_id in original_fisher:
        for name in original_fisher[block_id]:
            original_fisher[block_id][name] /= total_samples
    
    original_perplexity_avg = sum(original_perplexitys) / len(original_perplexitys)
    logger.info(f"Original_perplexity : {original_perplexity_avg}")

    # 2. Perturbation Loop
    test_bits = args.test_bit
    for bit in test_bits:
        est_fisher_info[bit] = {}
        modified_perplexitys[bit] = {}
        modified_fisher[bit] = {}
        
        block_number = 0
        # Iterate over transformer blocks
        for k, model_part in enumerate(model_parts):
            if k == 0 or k == len(model_parts) - 1:
                continue
            
            block_iterator = tqdm(model_part, desc=f"Block Processing (Bit {bit})")
            for seq_layer in block_iterator:
                modified_fisher[bit][block_number] = {}
                modified_perplexitys[bit][block_number] = {}
                target_layers = find_layers(seq_layer)
                
                count_layer = 0
                delta_thetas = {}
                param_list = {}
                
                # Perturb parameters
                for name, param in seq_layer.named_parameters():
                    if '.' in name:
                        layer_name, suffix = name.rsplit('.', 1)
                    else: continue
                    
                    if layer_name not in target_layers.keys() or 'bias' in suffix:
                        continue
                    
                    count_layer += 1
                    with torch.no_grad():
                         delta_theta = Quantization_perturbation(bit, target_layers[layer_name])
                         param.add_(delta_theta)
                         delta_thetas[layer_name] = delta_theta
                         param_list[layer_name] = param
                    param.requires_grad = True

                    # Run Forward/Backward for THIS perturbation batch
                    if count_layer % layer_batch == 0 or count_layer == len(target_layers):
                         # Re-run dataset
                         for step, batch in enumerate(dataset): # iterate dataset again
                            if isinstance(batch, dict):
                                input_ids = batch['input_ids'].to('cuda:0')
                            else:
                                input_ids = batch[0].to('cuda:0') if isinstance(batch, (tuple, list)) else batch.to('cuda:0')
                            if len(input_ids.shape) == 1: input_ids = input_ids.unsqueeze(0)
                            
                            # Just run one sample/batch to save time if needed, but original runs all
                            # We stick to original logic
                            batch_num_inner = input_ids.size(0)
                            for i in range(batch_num_inner):
                                inputs = input_ids[i].unsqueeze(0)
                                inp_kwargs = {
                                    "attention_mask": None,
                                    "position_ids": create_position_ids(len(input_ids[i]), device=args.device)
                                }
                                outputs = inputs
                                for j, mp in enumerate(model_parts):
                                    if j == 0: 
                                        device='cuda:0'
                                        outputs=mp(outputs.to(device))
                                        # Compute RoPE MANUALLY if needed
                                        if "position_embeddings" not in inp_kwargs:
                                            try:
                                                # Manually compute cos/sin
                                                seq_len = outputs.shape[1]
                                                cos, sin = compute_manual_rope(seq_len, head_dim, outputs.device)
                                                cos = cos.to(dtype=outputs.dtype)
                                                sin = sin.to(dtype=outputs.dtype)
                                                inp_kwargs["position_embeddings"] = (cos, sin)
                                            except: pass
                                    elif j == len(model_parts)-1: device=f'cuda:{num_gpus-1}' if num_gpus>0 else 'cpu'; outputs=mp(outputs.to(device))
                                    else: 
                                        device=f'cuda:{j%num_gpus}' if num_gpus>0 else 'cpu'
                                        for item in inp_kwargs:
                                            val = inp_kwargs[item]
                                            if val is not None:
                                                if isinstance(val, tuple): inp_kwargs[item] = tuple(v.to(device) for v in val)
                                                elif isinstance(val, torch.Tensor): inp_kwargs[item] = val.to(device)
                                        for l in mp: outputs=l(outputs.to(device), **inp_kwargs)[0]
                                
                                model.lm_head.to(f'cuda:{num_gpus-1}' if num_gpus>0 else 'cpu')
                                logits = model.lm_head(outputs)
                                inputs = inputs.to(f'cuda:{num_gpus-1}' if num_gpus>0 else 'cpu')
                                
                                loss_fn = nn.CrossEntropyLoss()
                                loss = loss_fn(logits[:, :-1, :].contiguous().view(-1, logits.size(-1)), inputs[:, 1:].view(-1))
                                
                                # Accumulate Loss for Modified Perplexity
                                for ln in param_list.keys():
                                    if ln not in modified_perplexitys[bit][block_number]:
                                        modified_perplexitys[bit][block_number][ln] = loss.item()
                                    else:
                                        modified_perplexitys[bit][block_number][ln] += loss.item()

                                # Backward
                                for mp in model_parts: mp.zero_grad()
                                loss.backward()
                                
                                # Accumulate Modified Fisher
                                for ln in param_list.keys():
                                    if param_list[ln].grad is not None:
                                        g2 = torch.sum(param_list[ln].grad.detach()**2).to("cpu")
                                        if ln not in modified_fisher[bit][block_number]:
                                            modified_fisher[bit][block_number][ln] = g2
                                        else:
                                            modified_fisher[bit][block_number][ln] += g2
                                
                                # Cleanup
                                for mp in model_parts: mp.zero_grad()
                                del inputs, outputs, logits, loss
                                torch.cuda.empty_cache()
                         
                         # Restore Params
                         with torch.no_grad():
                             for key in delta_thetas:
                                 param_list[key].sub_(delta_thetas[key])
                         delta_thetas = {}
                         param_list = {}
                         torch.cuda.empty_cache()

                block_number += 1

        # Normalize
        for bid in modified_fisher[bit]:
            for name in modified_fisher[bit][bid]:
                modified_fisher[bit][bid][name] /= total_samples
                modified_perplexitys[bit][bid][name] /= total_samples

        # Calculate Delta
        for bid in modified_fisher[bit]:
            est_fisher_info[bit][bid] = {}
            for name in modified_fisher[bit][bid]:
                est_fisher_info[bit][bid][name] = modified_fisher[bit][bid][name] - original_fisher[bid][name]

    return est_fisher_info, modified_perplexitys, original_perplexity_avg

# --- MAIN ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--test_bit", type=str, default="4")
    parser.add_argument("--nsamples", type=int, default=16)
    parser.add_argument("--seqlen", type=int, default=2048)
    args = parser.parse_args()
    
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    # Mock meta
    if "qwen" in args.model_path.lower():
        args.meta = MODEL_META["qwen"]
    else:
        args.meta = MODEL_META["llama"]
        
    args.test_bit = [int(x) for x in args.test_bit.split(",")]
    
    # Load Model
    logger.info(f"Loading model from {args.model_path}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
        
    # Mock Dataset (Random for standalone test)
    logger.info("Preparing dataset...")
    # Generate random input_ids
    dataset = []
    for _ in range(args.nsamples):
        dataset.append({'input_ids': torch.randint(0, 32000, (1, args.seqlen))})
    
    logger.info("Starting Fisher Evaluation...")
    fisher_info, modified_ppl, orig_ppl = evaluate_fisher_information_with_quant_perturb_sub_block(model, dataset, args)
    
    save_data(fisher_info, os.path.join(args.output_dir, "fisher_info.json"))
    save_data(modified_ppl, os.path.join(args.output_dir, "modified_ppl.json"))
    
    logger.info("Done.")