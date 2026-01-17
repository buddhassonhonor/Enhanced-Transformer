import argparse
import inspect
import json
import math
import os
import pickle
import time
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


# --- BEGIN model.py ---
class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Multi-Band Residual Fusion with Per-Head Gating
        hs = self.n_embd // self.n_head
        # Using Conv1d for filtering along the sequence dimension T.
        # Input to conv will be (B*nh, hs, T), so we use groups=hs for per-channel filtering.
        # Causal convolution is implemented by setting padding=0 and manually padding on the left in the forward pass.
        self.num_filters = 6
        self.filters = nn.ModuleList([
            nn.Conv1d(hs, hs, kernel_size=5, padding=0, bias=False) for _ in range(self.num_filters)
        ])

        # Learnable gates for fusing filter outputs, per head
        self.fusion_gates = nn.Parameter(torch.ones(1, self.n_head, 1, 1, self.num_filters))

        # Learnable per-head gating on the attention branch output
        self.head_gate = nn.Parameter(torch.ones(1, self.n_head, 1, hs))
        
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # Multi-Band Residual Fusion with Per-Head Gating
        y_attn_output = y
        B, nh, T, hs = y_attn_output.shape
        # Reshape for 1D convolution: (B, nh, T, hs) -> (B*nh, T, hs) -> (B*nh, hs, T)
        y_conv_in = y_attn_output.reshape(B * nh, T, hs).transpose(1, 2).contiguous()

        # Apply causal padding for Conv1d with kernel_size=5
        padding_size = self.filters[0].kernel_size[0] - 1
        y_conv_in_padded = F.pad(y_conv_in, (padding_size, 0))

        # Apply filters and collect outputs
        filtered_outputs = []
        for f in self.filters:
            y_filtered_conv = f(y_conv_in_padded)
            y_filtered_reshaped = y_filtered_conv.transpose(1, 2).view(B, nh, T, hs)
            filtered_outputs.append(y_filtered_reshaped)

        # Stack filtered outputs for gating
        y_filtered = torch.stack(filtered_outputs, dim=-1)  # (B, nh, T, hs, num_filters)

        # Apply learnable fusion gates
        fusion_gates = F.softmax(self.fusion_gates, dim=-1)
        y_fused = (y_filtered * fusion_gates).sum(dim=-1)  # (B, nh, T, hs)

        # Apply per-head output gating and add as residual
        y = y_attn_output + y_fused * torch.sigmoid(self.head_gate)

        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024  # GPT-2 standard context length
    vocab_size: int = 50257  # GPT-2 standard vocab size for subword tokenization
    n_layer: int = 12  # GPT-2 Small: 12 layers
    n_head: int = 12   # GPT-2 Small: 12 attention heads
    n_embd: int = 768  # GPT-2 Small: 768 embedding dimension
    dropout: float = 0.1  # GPT-2 Small standard dropout
    bias: bool = (
        True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    )


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert (
                t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size:]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# --- END model.py ---

def print_header():
    # Print a fun, academic-themed header
    print("="*80)
    print(f"ACADEMIC GPTWord 6L-6H-384 TRAINING")
    print("="*80)

def aggregate_and_summarize_runs(args):
    out_dir = args.out_dir
    all_results = []
    
    # Find all final_results files
    print(f"Looking for result files in {out_dir}...")
    for f in os.listdir(out_dir):
        if f.startswith(f"final_results_{args.dataset}_") and f.endswith(".json"):
            print(f"Found result file: {f}")
            with open(os.path.join(out_dir, f), "r") as fh:
                final_info = json.load(fh)
                # also load corresponding epoch data
                epoch_log_file = os.path.join(out_dir, f.replace("final_results_", "epoch_log_"))
                batch_log_file = os.path.join(out_dir, f.replace("final_results_", "batch_log_"))
                epoch_info = []
                batch_info = []
                if os.path.exists(epoch_log_file):
                    with open(epoch_log_file, 'r') as ef:
                        try:
                            # Try to load as a JSON array first
                            epoch_info = json.load(ef)
                        except json.JSONDecodeError:
                            # If that fails, try JSONL format (line by line)
                            ef.seek(0)  # Reset file pointer
                            epoch_info = []
                            for line in ef:
                                line = line.strip()
                                if line:  # Skip empty lines
                                    try:
                                        epoch_info.append(json.loads(line))
                                    except json.JSONDecodeError:
                                        print(f"Warning: Skipping invalid JSON line in {epoch_log_file}")
                if os.path.exists(batch_log_file):
                     with open(batch_log_file, 'r') as bf:
                        try:
                            # Try to load as a JSON array first
                            batch_info = json.load(bf)
                        except json.JSONDecodeError:
                            # If that fails, try JSONL format (line by line)
                            bf.seek(0)  # Reset file pointer
                            batch_info = []
                            for line in bf:
                                line = line.strip()
                                if line:  # Skip empty lines
                                    try:
                                        batch_info.append(json.loads(line))
                                    except json.JSONDecodeError:
                                        print(f"Warning: Skipping invalid JSON line in {batch_log_file}")
                
                all_results.append({
                    "seed": final_info.get("seed"),
                    "final_info": final_info,
                    "epoch_info": epoch_info,
                    "batch_info": batch_info
                })
    
    if not all_results:
        print(f"No result files found in {out_dir} for dataset {args.dataset}.")
        print("Make sure you have run training first or check the output directory.")
        return
    
    print(f"Found {len(all_results)} result files for aggregation.")
    
    # Calculate aggregate statistics for academic reporting
    val_losses = []
    for result in all_results:
        epoch_info = result["epoch_info"]
        if epoch_info:
            # Correctly extract the best validation loss from the epoch logs
            best_val_loss = min(e.get("val_loss", float('inf')) for e in epoch_info)
            if best_val_loss != float('inf'):
                val_losses.append(best_val_loss)

    train_losses = []
    for result in all_results:
        epoch_info = result["epoch_info"]
        if epoch_info:
            # Use the final training loss from the last epoch
            final_train_loss = epoch_info[-1].get("train_loss_avg", float('nan'))
            if not math.isnan(final_train_loss):
                train_losses.append(final_train_loss)

    training_times = [r["final_info"]["total_train_time"] for r in all_results]
    inference_speeds = [r["final_info"].get("avg_inference_tokens_per_second", 0) for r in all_results]
    
    # Extract best perplexity values from epoch logs (academic standard)
    val_perplexities = []
    for result in all_results:
        epoch_info = result["epoch_info"]
        if epoch_info:  # Check if epoch_info is not empty
            best_perplexity = min(e.get("val_perplexity", float('inf')) for e in epoch_info)
            if best_perplexity != float('inf'):
                val_perplexities.append(best_perplexity)
        else:
            # Fallback for older log formats or incomplete runs
            if "best_val_loss" in result["final_info"]:
                 val_perplexities.append(math.exp(result["final_info"]["best_val_loss"]))

    # Find the best validation loss from all runs
    best_val_loss_overall = float('inf')
    if val_losses:
        best_val_loss_overall = min(val_losses)

    summary = {
        "dataset": args.dataset,
        "num_seeds": len(all_results),  # Use actual number of results found
        "model_config": all_results[0]["final_info"]["config"],
        "aggregate_results": {
            "validation_loss": {
                "mean": float(np.mean(val_losses)) if val_losses else 0,
                "std": float(np.std(val_losses)) if val_losses else 0,
                "stderr": float(np.std(val_losses) / np.sqrt(len(val_losses))) if val_losses else 0,
                "individual": val_losses,
                "best": best_val_loss_overall
            },
            "validation_perplexity": {
                "mean": float(np.mean(val_perplexities)) if val_perplexities else 0,
                "std": float(np.std(val_perplexities)) if val_perplexities else 0,
                "stderr": float(np.std(val_perplexities) / np.sqrt(len(val_perplexities))) if val_perplexities else 0,
                "individual": val_perplexities
            },
            "training_loss": {
                "mean": float(np.mean(train_losses)) if train_losses else 0,
                "std": float(np.std(train_losses)) if train_losses else 0,
                "stderr": float(np.std(train_losses) / np.sqrt(len(train_losses))) if train_losses else 0,
                "individual": train_losses
            },
            "training_time_minutes": {
                "mean": float(np.mean(training_times) / 60) if training_times else 0,
                "std": float(np.std(training_times) / 60) if training_times else 0,
                "stderr": float(np.std(training_times) / np.sqrt(len(training_times)) / 60) if training_times else 0,
                "individual": [t/60 for t in training_times]
            },
            "inference_speed": {
                "mean": float(np.mean(inference_speeds)) if inference_speeds else 0,
                "std": float(np.std(inference_speeds)) if inference_speeds else 0,
                "stderr": float(np.std(inference_speeds) / np.sqrt(len(inference_speeds))) if inference_speeds else 0,
                "individual": inference_speeds
            }
        },
        "individual_runs": all_results
    }
    
    # Save comprehensive academic results
    with open(os.path.join(out_dir, f"academic_summary_{args.dataset}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    # Additionally, write nanoGPT-compatible final_info.json so AI-Scientist core can parse the run_x outputs
    final_info_json = {
        args.dataset: {
            "means": {
                "final_train_loss_mean": summary["aggregate_results"]["training_loss"]["mean"],
                "best_val_loss_mean": summary["aggregate_results"]["validation_loss"]["mean"],
                "perplexity_mean": summary["aggregate_results"]["validation_perplexity"]["mean"],
                "total_train_time_mean": summary["aggregate_results"]["training_time_minutes"]["mean"] * 60,
                "avg_inference_tokens_per_second_mean": summary["aggregate_results"]["inference_speed"]["mean"],
            },
            "stderrs": {
                "final_train_loss_stderr": summary["aggregate_results"]["training_loss"]["stderr"],
                "best_val_loss_stderr": summary["aggregate_results"]["validation_loss"]["stderr"],
                "perplexity_stderr": summary["aggregate_results"]["validation_perplexity"]["stderr"],
                "total_train_time_stderr": summary["aggregate_results"]["training_time_minutes"]["stderr"] * 60,
                "avg_inference_tokens_per_second_stderr": summary["aggregate_results"]["inference_speed"]["stderr"],
            },
            "final_info_dict": {
                "final_train_loss": summary["aggregate_results"]["training_loss"]["individual"],
                "best_val_loss": summary["aggregate_results"]["validation_loss"]["individual"],
                "perplexity": summary["aggregate_results"]["validation_perplexity"]["individual"],
                "total_train_time": [t * 60 for t in summary["aggregate_results"]["training_time_minutes"]["individual"]],
                "avg_inference_tokens_per_second": summary["aggregate_results"]["inference_speed"]["individual"],
            },
        }
    }

    # Load existing final_info.json if it exists, then merge with new results
    final_info_path = os.path.join(out_dir, "final_info.json")
    if os.path.exists(final_info_path):
        try:
            with open(final_info_path, "r", encoding="utf-8") as f:
                existing_final_info = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            existing_final_info = {}
    else:
        existing_final_info = {}
    
    # Add new dataset results to existing data
    existing_final_info.update(final_info_json)

    with open(final_info_path, "w", encoding="utf-8") as f:
        json.dump(existing_final_info, f, indent=2)
    
    # Print final academic summary
    print_header()
    print(f"Model: GPTWord Tiny ({all_results[0]['final_info']['model_params']:,} parameters)")
    print(f"Training: {all_results[0]['final_info']['num_epochs']} epochs")
    print(f"Statistical significance: {len(all_results)} seeds")
    print(f"\nACADEMIC METRICS:")
    print(f"Validation Perplexity: {summary['aggregate_results']['validation_perplexity']['mean']:.2f} ± {summary['aggregate_results']['validation_perplexity']['stderr']:.2f}")
    print(f"Best Validation Loss: {summary['aggregate_results']['validation_loss']['best']:.4f}")
    print(f"Average Validation Loss: {summary['aggregate_results']['validation_loss']['mean']:.4f} ± {summary['aggregate_results']['validation_loss']['stderr']:.4f}")
    print(f"Average Training Loss: {summary['aggregate_results']['training_loss']['mean']:.4f} ± {summary['aggregate_results']['training_loss']['stderr']:.4f}")
    print(f"Training Time: {summary['aggregate_results']['training_time_minutes']['mean']:.1f} ± {summary['aggregate_results']['training_time_minutes']['stderr']:.1f} minutes")
    print(f"Inference Speed: {summary['aggregate_results']['inference_speed']['mean']:.1f} ± {summary['aggregate_results']['inference_speed']['stderr']:.1f} tokens/sec")
    print(f"\nResults saved to: {out_dir}/academic_summary_{args.dataset}.json")
    print("Academic training completed successfully!")

def train(dataset="ptb", out_dir="run_0", seed_offset=0):
    """
    Academic training function using EPOCHS instead of iterations
    This ensures fair comparison across different methods and datasets
    """
    # -----------------------------------------------------------------------------
    # Academic training configuration - using EPOCHS for fair comparison
    
    # Dataset-specific configuration
    dataset_configs = {
        "wikitext2": {
            "num_epochs": 30,        # Consistent with PTB for fair comparison
            "batch_size": 16,
            "block_size": 1024,      # GPT-2 standard context
            "learning_rate": 3e-4,
            "is_char_level": False,
        },
        "wikitext103": {
            "num_epochs": 3,         # Fewer epochs due to larger dataset
            "batch_size": 8,         # Smaller batch due to memory
            "block_size": 1024,
            "learning_rate": 2e-4,
            "is_char_level": False,
        },
        "ptb": {
            "num_epochs": 30,        # Reduced for faster validation vs. baseline
            "batch_size": 16,
            "block_size": 1024,
            "learning_rate": 3e-4,
            "is_char_level": False,
        }
    }
    
    config = dataset_configs[dataset]
    
    # Training parameters
    gradient_accumulation_steps = 1
    batch_size = config["batch_size"]
    block_size = config["block_size"]
    num_epochs = config["num_epochs"]
    learning_rate = config["learning_rate"]
    is_char_level = config["is_char_level"]
    
    # Evaluation settings
    eval_interval_epochs = 1  # Evaluate every N epochs
    eval_iters = 100  # Number of batches for evaluation
    
    # Early stopping settings
    early_stopping_patience = 5  # Stop if no improvement for N epochs
    early_stopping_min_delta = 1e-4  # Minimum improvement threshold
    
    # Model - compact GPTWord configuration (approx 14M parameters)
    n_layer = 6   # Reduced layers for faster validation
    n_head = 6    # Fewer attention heads
    n_embd = 384  # Smaller embedding dimension
    dropout = 0.1  # standard for GPT-2 Small
    bias = True  # GPT-2 uses bias in layers
    weight_decay = 0.1  # GPT-2 standard
    beta1 = 0.9
    beta2 = 0.95  # GPT-2 standard
    grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings - now based on EPOCHS
    decay_lr = True  # whether to decay the learning rate
    warmup_epochs = 2  # warmup for first 2 epochs
    min_lr = learning_rate * 0.1  # 10% of max LR
    # DDP settings
    backend = "nccl"  # 'nccl', 'gloo', etc.
    # system
    device = "cuda"  # Always use CUDA
    dtype = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile = True  # RE-ENABLE torch.compile for performance

    # various inits, derived attributes, I/O setup
    master_process = True
    tokens_per_batch = gradient_accumulation_steps * batch_size * block_size
    
    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = "cuda" if "cuda" in device else "cpu"
    
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    # Use absolute path to data directory
    data_dir = os.path.join("d:\\", "data", dataset)
    
    # Calculate dataset statistics for EPOCH-based training
    print(f"Loading dataset: {dataset}")
    train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(data_dir, "validation.bin"), dtype=np.uint16, mode="r")
    
    # Calculate samples and batches per epoch
    train_samples = len(train_data) // block_size
    batches_per_epoch = train_samples // batch_size
    total_batches = batches_per_epoch * num_epochs
    
    print(f"Training configuration:")
    print(f"  Dataset: {dataset}")
    print(f"  Training tokens: {len(train_data):,}")
    print(f"  Validation tokens: {len(val_data):,}")
    print(f"  Samples per epoch: {train_samples:,}")
    print(f"  Batches per epoch: {batches_per_epoch:,}")
    print(f"  Total epochs: {num_epochs}")
    print(f"  Total batches: {total_batches:,}")
    print(f"  Tokens per batch: {tokens_per_batch:,}")
    print(f"  Total training tokens: {total_batches * tokens_per_batch:,}")

    def get_batch(split):
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        if split == "train":
            data = np.memmap(
                os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r"
            )
        else:
            data = np.memmap(
                os.path.join(data_dir, "validation.bin"), dtype=np.uint16, mode="r"
            )
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack(
            [torch.from_numpy((data[i: i + block_size]).astype(np.int64)) for i in ix]
        )
        y = torch.stack(
            [
                torch.from_numpy((data[i + 1: i + 1 + block_size]).astype(np.int64))
                for i in ix
            ]
        )
        
        if device_type == "cuda":
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
                device, non_blocking=True
            )
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    iter_num = 0
    best_val_loss = 1e9

    # attempt to derive vocab_size from the dataset
    meta_path = os.path.join(data_dir, "meta.pkl")
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        meta_vocab_size = meta["vocab_size"]
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
        
    # model init
    model_args = dict(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        bias=bias,
        vocab_size=None,
        dropout=dropout,
    )  # start with model_args from command line
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print(
            "defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)"
        )
    model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    
    # DEBUG: Print the actual vocab_size being used
    print(f"DEBUG: Model vocab_size set to: {gptconf.vocab_size}")
    
    model = GPT(gptconf)
    # crop down the model block size if desired, using model surgery
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args["block_size"] = (
            block_size  # so that the checkpoint will have the right value
        )
    model.to(device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.amp.GradScaler(enabled=(dtype == "float16"))

    # optimizer
    optimizer = model.configure_optimizers(
        weight_decay, learning_rate, (beta1, beta2), device_type
    )
    checkpoint = None  # free up memory

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        """Calculate standard academic metrics for language modeling"""
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            total_tokens = 0
            total_correct = 0
            
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                
                losses[k] = loss.item()
                
                # Calculate additional metrics
                predictions = torch.argmax(logits, dim=-1)
                correct = (predictions == Y).sum().item()
                total_correct += correct
                total_tokens += Y.numel()
            
            avg_loss = losses.mean()
            perplexity = torch.exp(avg_loss).item()
            accuracy = total_correct / total_tokens
            
            out[split] = {
                "loss": avg_loss,
                "perplexity": perplexity,
                "accuracy": accuracy
            }
        model.train()
        return out

    # learning rate decay scheduler (cosine with warmup) - EPOCH BASED
    def get_lr(epoch):
        """Cosine decay with linear warm-up that starts from min_lr, avoiding zero LR."""
        # 1) linear warm-up for the first <warmup_epochs> epochs.
        #    Start at `min_lr` for epoch 0 and increase to `learning_rate`.
        if epoch < warmup_epochs:
            warmup_ratio = (epoch + 1) / warmup_epochs  # epoch 0 -> 1/warmup_epochs
            return min_lr + warmup_ratio * (learning_rate - min_lr)
        # 2) if epoch > num_epochs, return min learning rate
        if epoch >= num_epochs:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)

    # logging - academic style with epoch tracking
    epoch_log_info = []
    batch_log_info = []
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    # EPOCH-BASED training loop
    print(f"\nStarting academic training for {num_epochs} epochs...")
    og_t0 = time.time()
    
    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"EPOCH {epoch + 1}/{num_epochs}")
        print(f"{'='*50}")
        
        model.train()
        epoch_start_time = time.time()
        epoch_train_loss = 0.0
        batch_count = 0
        
        # determine and set the learning rate for this epoch
        lr = get_lr(epoch) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        
        print(f"Learning rate: {lr:.2e}")
        
        # Training for one epoch
        for batch_idx in range(batches_per_epoch):
            X, Y = get_batch("train")
            
            # forward backward update
            optimizer.zero_grad()
            
            with ctx:
                logits, loss = model(X, Y)
            
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
            
            # clip the gradient
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            # step the optimizer
            scaler.step(optimizer)
            scaler.update()
            
            epoch_train_loss += loss.item()
            batch_count += 1
            
            # Log batch details for plotting
            batch_log_info.append({
                "epoch": epoch + 1,
                "batch": batch_idx,
                "loss": loss.item(),
                "lr": lr,
                "global_step": epoch * batches_per_epoch + batch_idx
            })
            
            # Print progress every 100 batches
            if batch_idx % 100 == 0 and batch_idx > 0:
                avg_loss = epoch_train_loss / batch_count
                elapsed = time.time() - epoch_start_time
                print(f"  Batch {batch_idx}/{batches_per_epoch}, Loss: {loss.item():.4f}, Avg: {avg_loss:.4f}, Time: {elapsed:.1f}s")
        
        # Calculate average training loss for this epoch
        avg_train_loss = epoch_train_loss / batch_count
        epoch_time = time.time() - epoch_start_time
        
        # Validation evaluation
        if epoch % eval_interval_epochs == 0 or epoch == num_epochs - 1:
             print(f"Evaluating with academic metrics...")
             metrics = estimate_loss()
             
             # Extract metrics
             val_loss = metrics["val"]["loss"].item()
             val_perplexity = metrics["val"]["perplexity"]
             val_accuracy = metrics["val"]["accuracy"]
             train_loss_eval = metrics["train"]["loss"].item()
             train_perplexity = metrics["train"]["perplexity"]
             train_accuracy = metrics["train"]["accuracy"]
             
             print(f"Epoch {epoch + 1} Results:")
             print(f"  Training Loss (avg): {avg_train_loss:.4f}")
             print(f"  Training Loss (eval): {train_loss_eval:.4f} | PPL: {train_perplexity:.2f} | Acc: {train_accuracy:.3f}")
             print(f"  Validation Loss: {val_loss:.4f} | PPL: {val_perplexity:.2f} | Acc: {val_accuracy:.3f}")
             print(f"  Epoch Time: {epoch_time:.1f}s")
             
             # Record epoch results with academic metrics
             epoch_log_info.append({
                 "epoch": epoch + 1,
                 "train_loss_avg": avg_train_loss,
                 "train_loss_eval": train_loss_eval,
                 "train_perplexity": train_perplexity,
                 "train_accuracy": train_accuracy,
                 "val_loss": val_loss,
                 "val_perplexity": val_perplexity,
                 "val_accuracy": val_accuracy,
                 "lr": lr,
                 "epoch_time": epoch_time,
                 "batches": batch_count
             })
             
             # Save best model and early stopping logic
             if val_loss < best_val_loss - early_stopping_min_delta:
                 best_val_loss = val_loss
                 epochs_without_improvement = 0
                 print(f"  *** New best validation loss: {best_val_loss:.4f}")
                 
                 # Model saving disabled to save storage space
             else:
                 epochs_without_improvement += 1
                 print(f"  No improvement for {epochs_without_improvement} epochs")
                 
             # Early stopping check
             if epochs_without_improvement >= early_stopping_patience:
                 print(f"  Early stopping triggered after {epochs_without_improvement} epochs without improvement")
                 break
        else:
            print(f"Epoch {epoch + 1} - Training Loss: {avg_train_loss:.4f}, Time: {epoch_time:.1f}s")

    total_time = time.time() - og_t0
    print(f"\n{'='*50}")
    print("ACADEMIC TRAINING COMPLETED")
    print(f"{'='*50}")
    print(f"Dataset: {dataset}")
    print(f"Total epochs: {num_epochs}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Total training time: {total_time / 60:.2f} minutes")
    print(f"Total training time: {total_time / 3600:.2f} hours")
    print(f"Average time per epoch: {total_time / num_epochs:.1f} seconds")

    # Calculate best validation perplexity
    best_val_perplexity = math.exp(best_val_loss)
    
    final_info = {
        "dataset": dataset,
        "num_epochs": num_epochs,
        "final_train_loss": avg_train_loss,
        "best_val_loss": best_val_loss,
        "best_val_perplexity": best_val_perplexity,
        "total_train_time": total_time,
        "batches_per_epoch": batches_per_epoch,
        "total_batches": total_batches,
        "model_params": model.get_num_params(),
        "config": config
    }

    # === SAMPLING SCRIPT ===

    # New parameters for generation
    start = " "
    num_samples = 10  # number of samples to draw
    max_new_tokens = 500  # number of tokens generated in each sample
    temperature = (
        0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    )
    top_k = 200  # retain only the top_k most likely tokens, clamp others to have 0 probability

    # Encoding setup - now only subword tokenization is needed
    print(f"Setting up tokenization for {dataset}...")
    
    # Subword tokenization (wikitext2, wikitext103, ptb) - USE TIKTOKEN LOCALLY
    print("Using local tiktoken GPT-2 tokenizer to avoid network errors.")
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s)
    decode = lambda l: enc.decode(l)
    print(f"Using GPT-2 subword tokenization via tiktoken, vocab size: {enc.n_vocab}")

    # Encode the beginning of the prompt
    if start.startswith("FILE:"):
        with open(start[5:], "r", encoding="utf-8") as f:
            start = f.read()
    start_ids = encode(start)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    # Run generation
    model.eval()
    results = []
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                start_time = time.time()
                y = model.generate(
                    x, max_new_tokens, temperature=temperature, top_k=top_k
                )
                end_time = time.time()

                generated_text = decode(y[0].tolist())
                inference_time = end_time - start_time
                tokens_per_second = max_new_tokens / inference_time

                print(f"Sample {k + 1}:")
                print(generated_text)
                print(f"Inference time: {inference_time:.2f} seconds")
                print(f"Tokens per second: {tokens_per_second:.2f}")
                print("---------------")

                results.append(
                    {
                        "sample_id": k + 1,
                        "generated_text": generated_text,
                        "inference_time": inference_time,
                        "tokens_per_second": tokens_per_second,
                    }
                )

    # Calculate and print average inference speed
    avg_tokens_per_second = sum(r["tokens_per_second"] for r in results) / len(results)
    print(f"Average tokens per second: {avg_tokens_per_second:.2f}")

    final_info["avg_inference_tokens_per_second"] = avg_tokens_per_second

    # Save comprehensive results for academic analysis
    with open(os.path.join(out_dir, f"final_results_{dataset}_{seed_offset}.json"), "w") as f:
        json.dump(final_info, f, indent=2)
    
    with open(os.path.join(out_dir, f"epoch_log_{dataset}_{seed_offset}.json"), "w") as f:
        json.dump(epoch_log_info, f, indent=2)
    
    with open(os.path.join(out_dir, f"batch_log_{dataset}_{seed_offset}.json"), "w") as f:
        json.dump(batch_log_info, f, indent=2)
    
    print(f"\nResults saved to {out_dir}/")
    print("Files for academic analysis:")
    print(f"  - final_results_{dataset}_{seed_offset}.json")
    print(f"  - epoch_log_{dataset}_{seed_offset}.json") 
    print(f"  - batch_log_{dataset}_{seed_offset}.json")
    
    return final_info, epoch_log_info, batch_log_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment")
    
    # Academic experimental setup - supporting multiple datasets
    parser.add_argument("--dataset", type=str, default="wikitext2", 
                       choices=["wikitext2", "wikitext103", "ptb"],
                       help="Dataset to train on")
    # Allow setting num_seeds in code like nanoGPT; command-line overrides
    parser.add_argument("--num_seeds", type=int, default=None, 
                       help="Number of random seeds for statistical significance")
    parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")


    args = parser.parse_args()
    
    print_header()
    print(f"Dataset: {args.dataset}")
    # ------------------------------------------------------------------
    # nanoGPT-style per-dataset seed defaults and idea-run override
    DEFAULT_NUM_SEEDS = {
        "wikitext2": 3,
        "wikitext103": 3,
        "ptb": 3,
    }

    # If user did not pass --num_seeds, fall back to dictionary
    if args.num_seeds is None:
        args.num_seeds = DEFAULT_NUM_SEEDS.get(args.dataset, 1)

    # For idea folders (non-baseline), use a single seed by convention
    # Commented out to allow multiple seeds for all runs
    # if args.out_dir != "run_0":
    #     args.num_seeds = 1

    print(f"Resolved num_seeds: {args.num_seeds}")

    print(f"Output directory: {args.out_dir}")
    
    # Run multiple seeds for statistical significance
    out_dir = args.out_dir
    all_results = []
    
    # If num_seeds is 0, skip training and only run aggregation
    if args.num_seeds == 0:
        print("Skipping training (num_seeds=0), running aggregation only...")
        aggregate_and_summarize_runs(args)
    else:
        for seed_offset in range(args.num_seeds):
            print(f"\n{'='*60}")
            print(f"RUNNING SEED {seed_offset + 1}/{args.num_seeds}")
            print(f"{'='*60}")
            
            final_info, epoch_info, batch_info = train(
                dataset=args.dataset, 
                out_dir=args.out_dir, 
                seed_offset=seed_offset
            )
            all_results.append({
                "seed": seed_offset,
                "final_info": final_info,
                "epoch_info": epoch_info,
                "batch_info": batch_info
            })
        
        # Calculate aggregate statistics for academic reporting
        val_losses = []
        for result in all_results:
            epoch_info = result["epoch_info"]
            if epoch_info:
                # Correctly extract the best validation loss from the epoch logs
                best_val_loss = min(e.get("val_loss", float('inf')) for e in epoch_info)
                if best_val_loss != float('inf'):
                    val_losses.append(best_val_loss)

        train_losses = []
        for result in all_results:
            epoch_info = result["epoch_info"]
            if epoch_info:
                # Use the final training loss from the last epoch
                final_train_loss = epoch_info[-1].get("train_loss_avg", float('nan'))
                if not math.isnan(final_train_loss):
                    train_losses.append(final_train_loss)

        training_times = [r["final_info"]["total_train_time"] for r in all_results]
        inference_speeds = [r["final_info"].get("avg_inference_tokens_per_second", 0) for r in all_results]
        
        # Extract best perplexity values from epoch logs (academic standard)
        val_perplexities = []
        for result in all_results:
            epoch_info = result["epoch_info"]
            if epoch_info:  # Check if epoch_info is not empty
                best_perplexity = min(e.get("val_perplexity", float('inf')) for e in epoch_info)
                if best_perplexity != float('inf'):
                    val_perplexities.append(best_perplexity)
            else:
                # Fallback for older log formats or incomplete runs
                if "best_val_loss" in result["final_info"]:
                     val_perplexities.append(math.exp(result["final_info"]["best_val_loss"]))

        # Find the best validation loss from all runs
        best_val_loss_overall = float('inf')
        if val_losses:
            best_val_loss_overall = min(val_losses)

        summary = {
            "dataset": args.dataset,
            "num_seeds": args.num_seeds,
            "model_config": all_results[0]["final_info"]["config"],
            "aggregate_results": {
                "validation_loss": {
                    "mean": float(np.mean(val_losses)) if val_losses else 0,
                    "std": float(np.std(val_losses)) if val_losses else 0,
                    "stderr": float(np.std(val_losses) / np.sqrt(len(val_losses))) if val_losses else 0,
                    "individual": val_losses,
                    "best": best_val_loss_overall
                },
                "validation_perplexity": {
                    "mean": float(np.mean(val_perplexities)) if val_perplexities else 0,
                    "std": float(np.std(val_perplexities)) if val_perplexities else 0,
                    "stderr": float(np.std(val_perplexities) / np.sqrt(len(val_perplexities))) if val_perplexities else 0,
                    "individual": val_perplexities
                },
                "training_loss": {
                    "mean": float(np.mean(train_losses)) if train_losses else 0,
                    "std": float(np.std(train_losses)) if train_losses else 0,
                    "stderr": float(np.std(train_losses) / np.sqrt(len(train_losses))) if train_losses else 0,
                    "individual": train_losses
                },
                "training_time_minutes": {
                    "mean": float(np.mean(training_times) / 60),
                    "std": float(np.std(training_times) / 60),
                    "stderr": float(np.std(training_times) / np.sqrt(len(training_times)) / 60),
                    "individual": [t/60 for t in training_times]
                },
                "inference_speed": {
                    "mean": float(np.mean(inference_speeds)),
                    "std": float(np.std(inference_speeds)),
                    "stderr": float(np.std(inference_speeds) / np.sqrt(len(inference_speeds))),
                    "individual": inference_speeds
                }
            },
            "individual_runs": all_results
        }
        
        # Save comprehensive academic results
        with open(os.path.join(out_dir, f"academic_summary_{args.dataset}.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        
            # Also output nanoGPT-compatible final_info.json for AI-Scientist automation
    # Load existing final_info.json if it exists to merge results
    final_info_path = os.path.join(out_dir, "final_info.json")
    if os.path.exists(final_info_path):
        with open(final_info_path, "r", encoding="utf-8") as f:
            final_info_json = json.load(f)
    else:
        final_info_json = {}
    
    # Add/update current dataset results
    final_info_json[args.dataset] = {
        "means": {
            "final_train_loss_mean": summary["aggregate_results"]["training_loss"]["mean"],
            "best_val_loss_mean": summary["aggregate_results"]["validation_loss"]["mean"],
            "perplexity_mean": summary["aggregate_results"]["validation_perplexity"]["mean"],
            "total_train_time_mean": summary["aggregate_results"]["training_time_minutes"]["mean"] * 60,
            "avg_inference_tokens_per_second_mean": summary["aggregate_results"]["inference_speed"]["mean"],
        },
        "stderrs": {
            "final_train_loss_stderr": summary["aggregate_results"]["training_loss"]["stderr"],
            "best_val_loss_stderr": summary["aggregate_results"]["validation_loss"]["stderr"],
            "perplexity_stderr": summary["aggregate_results"]["validation_perplexity"]["stderr"],
            "total_train_time_stderr": summary["aggregate_results"]["training_time_minutes"]["stderr"] * 60,
            "avg_inference_tokens_per_second_stderr": summary["aggregate_results"]["inference_speed"]["stderr"],
        },
        "final_info_dict": {
            "final_train_loss": summary["aggregate_results"]["training_loss"]["individual"],
            "best_val_loss": summary["aggregate_results"]["validation_loss"]["individual"],
            "perplexity": summary["aggregate_results"]["validation_perplexity"]["individual"],
            "total_train_time": [t * 60 for t in summary["aggregate_results"]["training_time_minutes"]["individual"]],
            "avg_inference_tokens_per_second": summary["aggregate_results"]["inference_speed"]["individual"],
        },
    }

    with open(final_info_path, "w", encoding="utf-8") as f:
        json.dump(final_info_json, f, indent=2)
        
        # Print final academic summary
        print_header()
        print(f"Model: GPTWord Tiny ({all_results[0]['final_info']['model_params']:,} parameters)")
        print(f"Training: {all_results[0]['final_info']['num_epochs']} epochs")
        print(f"Statistical significance: {args.num_seeds} seeds")
        print(f"\nACADEMIC METRICS:")
        print(f"Validation Perplexity: {summary['aggregate_results']['validation_perplexity']['mean']:.2f} ± {summary['aggregate_results']['validation_perplexity']['stderr']:.2f}")
        print(f"Best Validation Loss: {summary['aggregate_results']['validation_loss']['best']:.4f}")
        print(f"Average Validation Loss: {summary['aggregate_results']['validation_loss']['mean']:.4f} ± {summary['aggregate_results']['validation_loss']['stderr']:.4f}")
        print(f"Average Training Loss: {summary['aggregate_results']['training_loss']['mean']:.4f} ± {summary['aggregate_results']['training_loss']['stderr']:.4f}")
        print(f"Training Time: {summary['aggregate_results']['training_time_minutes']['mean']:.1f} ± {summary['aggregate_results']['training_time_minutes']['stderr']:.1f} minutes")
        print(f"Inference Speed: {summary['aggregate_results']['inference_speed']['mean']:.1f} ± {summary['aggregate_results']['inference_speed']['stderr']:.1f} tokens/sec")
        print(f"\nResults saved to: {out_dir}/academic_summary_{args.dataset}.json")
        print("Academic training completed successfully!")
