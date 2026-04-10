import json
import logging
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.decomposition import FastICA, PCA
from transformer_lens import HookedTransformer

from sae_lens import SAE

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


@dataclass
class PipelineConfig:
    model_name: str = "google/gemma-2-2b"
    dataset_name: str = "NeelNanda/pile-10k"
    dataset_split: str = "train"
    output_dir: str = "results/concept_extraction_logs"
    seed: int = 42

    source_layer: int = 11
    target_layer: int = 12
    source_site: str = "resid_mid"
    target_site: str = "mlp_post"
    target_neuron: int = 42

    seq_len: int = 32
    seq_pos: int = 15

    bank_size: int = 10000
    bank_batch_size: int = 16
    num_runs: int = 10
    langevin_steps: int = 100
    learning_rate: float = 0.05
    prior_strength: float = 0.1
    kde_bandwidth_sigma2: float = 10.0
    noise_scale: float = 1.0

    convergence_fraction_of_best: float = 0.8
    minimum_success_count: int = 3
    max_pairwise_cosine_warning: float = 0.99

    pca_variance_threshold: float = 0.90
    max_concepts: int = 5
    fastica_max_iter: int = 2000
    semi_nmf_max_iter: int = 500
    semi_nmf_tol: float = 1e-5

    top_k_tokens: int = 15
    plot_outputs: bool = True

    sae_enabled: bool = True
    sae_path: str = "layer_11/width_16k/average_l0_79"
    sae_neuropedia_model: str = "gemma-2-2b"
    sae_neuropedia_id: str = "11-gemmascope-res-16k"

    dtype: str = "float16"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def setup_logging(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "pipeline.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True, 
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def get_source_hook_name(config: PipelineConfig) -> str:
    site_map = {
        "resid_pre": f"blocks.{config.source_layer}.hook_resid_pre",
        "resid_mid": f"blocks.{config.source_layer}.hook_resid_mid",
        "resid_post": f"blocks.{config.source_layer}.hook_resid_post",
    }
    if config.source_site not in site_map:
        raise ValueError(f"Unsupported source site: {config.source_site}")
    return site_map[config.source_site]


def get_target_hook_name(config: PipelineConfig) -> str:
    site_map = {
        "mlp_pre": f"blocks.{config.target_layer}.mlp.hook_pre",
        "mlp_post": f"blocks.{config.target_layer}.mlp.hook_post",
    }
    if config.target_site not in site_map:
        raise ValueError(f"Unsupported target site: {config.target_site}")
    return site_map[config.target_site]


def ensure_tokenizer_padding(model: HookedTransformer) -> None:
    tokenizer = model.tokenizer
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            raise ValueError("Tokenizer has no pad/eos/unk token available for fixed-length padding.")
    tokenizer.padding_side = "right"


def build_fixed_dummy_tokens(
    model: HookedTransformer,
    seq_len: int,
    device: str,
) -> torch.Tensor:
    tokenizer = model.tokenizer
    filler_id = tokenizer.pad_token_id
    if filler_id is None:
        filler_id = tokenizer.eos_token_id
    if filler_id is None:
        filler_id = tokenizer.unk_token_id
    if filler_id is None:
        raise ValueError("Tokenizer has no pad/eos/unk token available for dummy prompt construction.")

    dummy = torch.full((1, seq_len), filler_id, dtype=torch.long)
    bos_id = tokenizer.bos_token_id
    if bos_id is not None and seq_len > 0:
        dummy[0, 0] = bos_id
    return dummy.to(device)


def tokenize_text_batch(
    model: HookedTransformer,
    texts: List[str],
    seq_len: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    tokenized = model.tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        add_special_tokens=True,
    )
    tokens = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    return tokens, attention_mask


def collect_empirical_bank(
    model: HookedTransformer,
    dataset,
    config: PipelineConfig,
    source_hook_name: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if config.seq_pos >= config.seq_len:
        raise ValueError("seq_pos must be strictly less than seq_len.")

    collected: List[torch.Tensor] = []
    total_examined = 0
    total_kept = 0

    for start in range(0, len(dataset), config.bank_batch_size):
        if total_kept >= config.bank_size:
            break

        batch = dataset[start : start + config.bank_batch_size]["text"]
        total_examined += len(batch)
        tokens, attention_mask = tokenize_text_batch(model, batch, config.seq_len, config.device)

        valid_mask = attention_mask[:, config.seq_pos].bool()
        if not torch.any(valid_mask):
            continue

        valid_tokens = tokens[valid_mask]
        cache: Dict[str, torch.Tensor] = {}

        def cache_source_hook(resid: torch.Tensor, hook) -> torch.Tensor:
            cache["source"] = resid[:, config.seq_pos, :].detach()
            return resid

        with torch.no_grad():
            model.run_with_hooks(valid_tokens, fwd_hooks=[(source_hook_name, cache_source_hook)])

        if "source" not in cache:
            raise RuntimeError(f"Source hook {source_hook_name} did not fire.")

        bank_batch = cache["source"]
        collected.append(bank_batch.cpu())
        total_kept += bank_batch.shape[0]

        logging.info(
            "Collected %d/%d residual vectors for X_bank after examining %d texts.",
            min(total_kept, config.bank_size),
            config.bank_size,
            total_examined,
        )

    if not collected:
        raise RuntimeError(
            "No valid residual vectors were collected. Increase seq_len, lower seq_pos, or check tokenization."
        )

    x_bank = torch.cat(collected, dim=0)[: config.bank_size].contiguous()
    x_mean = x_bank.mean(dim=0, keepdim=True)
    return x_bank, x_mean


def run_activation_with_injection(
    model: HookedTransformer,
    dummy_tokens: torch.Tensor,
    source_hook_name: str,
    target_hook_name: str,
    seq_pos: int,
    target_neuron: int,
    injected_vector: torch.Tensor,
) -> torch.Tensor:
    activation_box: Dict[str, torch.Tensor] = {}

    def inject_hook(resid: torch.Tensor, hook) -> torch.Tensor:
        resid = resid.clone()
        resid[:, seq_pos, :] = injected_vector.to(resid.device)
        return resid

    def measure_hook(act_tensor: torch.Tensor, hook) -> torch.Tensor:
        activation_box["activation"] = act_tensor[0, seq_pos, target_neuron]
        return act_tensor

    model.run_with_hooks(
        dummy_tokens,
        fwd_hooks=[
            (source_hook_name, inject_hook),
            (target_hook_name, measure_hook),
        ],
    )

    if "activation" not in activation_box:
        raise RuntimeError(f"Target hook {target_hook_name} did not fire.")
    return activation_box["activation"]


def compute_kde_terms(
    x_t: torch.Tensor,
    x_bank: torch.Tensor,
    sigma2: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    diffs = x_bank - x_t
    sq_dists = torch.sum(diffs * diffs, dim=1)
    logits = -sq_dists / (2.0 * sigma2)
    weights = torch.softmax(logits, dim=0)
    center_of_mass = torch.sum(weights.unsqueeze(1) * x_bank, dim=0, keepdim=True)
    score = center_of_mass - x_t
    return score, weights, sq_dists


def kde_log_density(x_t: torch.Tensor, x_bank: torch.Tensor, sigma2: float) -> float:
    sq_dists = torch.sum((x_bank - x_t) ** 2, dim=1)
    d_model = x_bank.shape[1]
    logits = -sq_dists / (2.0 * sigma2)
    log_norm = 0.5 * d_model * math.log(2.0 * math.pi * sigma2)
    log_prob = torch.logsumexp(logits, dim=0) - math.log(x_bank.shape[0]) - log_norm
    return float(log_prob.item())


def langevin_optimize(
    model: HookedTransformer,
    dummy_tokens: torch.Tensor,
    x_bank: torch.Tensor,
    initial_vector: torch.Tensor,
    config: PipelineConfig,
    source_hook_name: str,
    target_hook_name: str,
) -> Dict[str, object]:
    x_t = initial_vector.clone().to(config.device)
    trace: List[float] = []

    for step in range(config.langevin_steps):
        x_t = x_t.detach().requires_grad_(True)
        activation = run_activation_with_injection(
            model=model,
            dummy_tokens=dummy_tokens,
            source_hook_name=source_hook_name,
            target_hook_name=target_hook_name,
            seq_pos=config.seq_pos,
            target_neuron=config.target_neuron,
            injected_vector=x_t,
        )
        
        # Calculate gradients ONLY for x_t instead of the entire model
        grad = torch.autograd.grad(outputs=activation, inputs=x_t)[0].detach()

        with torch.no_grad():
            score, weights, sq_dists = compute_kde_terms(
                x_t=x_t.detach(),
                x_bank=x_bank,
                sigma2=config.kde_bandwidth_sigma2,
            )
            noise = (
                math.sqrt(2.0 * config.learning_rate)
                * config.noise_scale
                * torch.randn_like(x_t)
            )
            x_t = (
                x_t.detach()
                + config.learning_rate * grad
                + (config.prior_strength * config.learning_rate / config.kde_bandwidth_sigma2) * score
                + noise
            )

        trace.append(float(activation.item()))

        if step % 10 == 0 or step == config.langevin_steps - 1:
            logging.info(
                "Run step %d/%d | activation=%.4f | grad_norm=%.4f | mean_dist=%.4f",
                step + 1,
                config.langevin_steps,
                activation.item(),
                grad.norm().item(),
                sq_dists.mean().item(),
            )

    final_activation = run_activation_with_injection(
        model=model,
        dummy_tokens=dummy_tokens,
        source_hook_name=source_hook_name,
        target_hook_name=target_hook_name,
        seq_pos=config.seq_pos,
        target_neuron=config.target_neuron,
        injected_vector=x_t.detach(),
    )

    return {
        "vector": x_t.detach().cpu(),
        "final_activation": float(final_activation.item()),
        "trace": trace,
        "kde_log_density": kde_log_density(
            x_t.detach(),
            x_bank=x_bank,
            sigma2=config.kde_bandwidth_sigma2,
        ),
    }


def compute_pairwise_cosine_stats(vectors: torch.Tensor) -> Dict[str, float]:
    if vectors.shape[0] < 2:
        return {"mean": float("nan"), "max": float("nan")}
    normalized = F.normalize(vectors, dim=1)
    cos = normalized @ normalized.T
    mask = ~torch.eye(cos.shape[0], dtype=torch.bool, device=cos.device)
    values = cos[mask]
    return {
        "mean": float(values.mean().item()),
        "max": float(values.max().item()),
    }


def choose_rank_from_pca(v_rows: np.ndarray, explained_variance: float, max_concepts: int) -> int:
    max_rank = min(v_rows.shape[0], v_rows.shape[1], max_concepts)
    if max_rank < 1:
        raise ValueError("Need at least one optimized vector to choose PCA rank.")
    pca = PCA(n_components=max_rank, random_state=0)
    pca.fit(v_rows)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cumulative, explained_variance) + 1)
    return max(1, min(k, max_rank))


def split_pos_neg(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pos = np.maximum(matrix, 0.0)
    neg = np.maximum(-matrix, 0.0)
    return pos, neg


def semi_nmf(
    v_cols: np.ndarray,
    rank: int,
    max_iter: int,
    tol: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    rng = np.random.default_rng(seed)
    d_model, num_vectors = v_cols.shape

    h = rng.random((rank, num_vectors), dtype=np.float64) + 1e-3
    hh_t = h @ h.T
    w = v_cols @ h.T @ np.linalg.pinv(hh_t)
    losses: List[float] = []

    eps = 1e-12
    for _ in range(max_iter):
        wt_v = w.T @ v_cols
        wt_wh = (w.T @ w) @ h
        wt_v_pos, wt_v_neg = split_pos_neg(wt_v)
        wt_wh_pos, wt_wh_neg = split_pos_neg(wt_wh)

        numerator = wt_v_pos + wt_wh_neg + eps
        denominator = wt_v_neg + wt_wh_pos + eps
        h *= np.sqrt(numerator / denominator)

        hh_t = h @ h.T
        w = v_cols @ h.T @ np.linalg.pinv(hh_t)

        recon = w @ h
        loss = float(np.linalg.norm(v_cols - recon, ord="fro"))
        losses.append(loss)

        if len(losses) > 1:
            rel_change = abs(losses[-2] - losses[-1]) / max(losses[-2], eps)
            if rel_change < tol:
                break

    return w, h, losses


def normalize_concepts(concepts: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(concepts, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return concepts / norms


def run_decomposition(
    successful_vectors: torch.Tensor,
    config: PipelineConfig,
) -> Dict[str, Dict[str, object]]:
    v_rows = successful_vectors.numpy()
    rank = choose_rank_from_pca(
        v_rows=v_rows,
        explained_variance=config.pca_variance_threshold,
        max_concepts=config.max_concepts,
    )
    logging.info("Selected decomposition rank k=%d from PCA threshold %.2f.", rank, config.pca_variance_threshold)

    results: Dict[str, Dict[str, object]] = {}

    ica = FastICA(
        n_components=rank,
        random_state=config.seed,
        max_iter=config.fastica_max_iter,
        whiten="unit-variance",
    )
    ica_scores = ica.fit_transform(v_rows)
    ica_components = ica.components_.copy()
    for i in range(rank):
        if float(np.mean(ica_scores[:, i])) < 0.0:
            ica_components[i] *= -1.0
            ica_scores[:, i] *= -1.0
    ica_components = normalize_concepts(ica_components)
    results["fastica"] = {
        "concepts": ica_components,
        "loadings": ica_scores,
        "rank": rank,
    }

    v_cols = v_rows.T
    semi_w, semi_h, semi_losses = semi_nmf(
        v_cols=v_cols,
        rank=rank,
        max_iter=config.semi_nmf_max_iter,
        tol=config.semi_nmf_tol,
        seed=config.seed,
    )
    semi_concepts = normalize_concepts(semi_w.T)
    results["semi_nmf"] = {
        "concepts": semi_concepts,
        "loadings": semi_h.T,
        "rank": rank,
        "loss_trace": semi_losses,
    }

    return results


def l2_restore(original: torch.Tensor, modified: torch.Tensor) -> torch.Tensor:
    original_norm = torch.norm(original)
    modified_norm = torch.norm(modified).clamp_min(1e-12)
    return modified * (original_norm / modified_norm)


def evaluate_concepts(
    model: HookedTransformer,
    dummy_tokens: torch.Tensor,
    x_bank: torch.Tensor,
    x_mean: torch.Tensor,
    successful_vectors: torch.Tensor,
    final_activations: torch.Tensor,
    decomp_result: Dict[str, object],
    config: PipelineConfig,
    source_hook_name: str,
    target_hook_name: str,
) -> List[Dict[str, object]]:
    target_dtype = successful_vectors.dtype
    concepts = torch.tensor(decomp_result["concepts"], dtype=target_dtype, device=config.device)
    loadings = torch.tensor(decomp_result["loadings"], dtype=target_dtype, device=config.device)
    
    successful_vectors_device = successful_vectors.to(config.device)
    x_mean_device = x_mean.to(config.device)
    x_bank_device = x_bank.to(config.device)

    baseline_activation = run_activation_with_injection(
        model=model,
        dummy_tokens=dummy_tokens,
        source_hook_name=source_hook_name,
        target_hook_name=target_hook_name,
        seq_pos=config.seq_pos,
        target_neuron=config.target_neuron,
        injected_vector=x_mean_device,
    )

    results: List[Dict[str, object]] = []

    for idx in range(concepts.shape[0]):
        c_k = concepts[idx : idx + 1]
        projected_activations = (successful_vectors_device @ c_k.T).squeeze(1)
        winner = int(torch.argmax(projected_activations).item())
        x_opt = successful_vectors_device[winner : winner + 1]

        cosine_precheck = F.cosine_similarity(x_opt, c_k, dim=1).item()

        scalar_projection = torch.sum(x_opt * c_k, dim=1, keepdim=True)
        x_ablated = x_opt - scalar_projection * c_k
        x_final_ablated = l2_restore(x_opt, x_ablated)

        a_opt = run_activation_with_injection(
            model=model,
            dummy_tokens=dummy_tokens,
            source_hook_name=source_hook_name,
            target_hook_name=target_hook_name,
            seq_pos=config.seq_pos,
            target_neuron=config.target_neuron,
            injected_vector=x_opt,
        )
        a_ablated = run_activation_with_injection(
            model=model,
            dummy_tokens=dummy_tokens,
            source_hook_name=source_hook_name,
            target_hook_name=target_hook_name,
            seq_pos=config.seq_pos,
            target_neuron=config.target_neuron,
            injected_vector=x_final_ablated,
        )

        denom = (a_opt - baseline_activation).item()
        drop_ratio = 0.0 if abs(denom) < 1e-9 else (a_opt.item() - a_ablated.item()) / denom

        kde_opt = kde_log_density(x_opt, x_bank_device, config.kde_bandwidth_sigma2)
        kde_ablated = kde_log_density(x_final_ablated, x_bank_device, config.kde_bandwidth_sigma2)

        alpha = float(projected_activations.mean().item())
        x_concept = x_mean_device + alpha * c_k
        off_manifold_warning = bool(kde_ablated < kde_opt - 5.0)

        results.append(
            {
                "concept_index": idx,
                "concept_vector": c_k.detach().cpu(),
                "winning_vector_index": winner,
                "winning_final_activation": float(final_activations[winner].item()),
                "mean_factor_loading": float(loadings[:, idx].mean().item()),
                "mean_projected_activation": alpha,
                "winner_projected_activation": float(projected_activations[winner].item()),
                "cosine_precheck": cosine_precheck,
                "activation_opt": float(a_opt.item()),
                "activation_ablated": float(a_ablated.item()),
                "activation_baseline": float(baseline_activation.item()),
                "drop_ratio": float(drop_ratio),
                "kde_log_density_opt": kde_opt,
                "kde_log_density_ablated": kde_ablated,
                "off_manifold_warning": off_manifold_warning,
                "alpha": alpha,
                "x_opt": x_opt.detach().cpu(),
                "x_ablated": x_final_ablated.detach().cpu(),
                "x_concept": x_concept.detach().cpu(),
            }
        )

    return results


def get_final_norm_module(model: HookedTransformer):
    if hasattr(model, "ln_final") and model.ln_final is not None:
        return model.ln_final
    if hasattr(model, "norm") and model.norm is not None:
        return model.norm
    raise AttributeError("Could not locate the model's final LayerNorm/RMSNorm module.")


def get_unembedding_weights(model: HookedTransformer) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if hasattr(model, "unembed") and hasattr(model.unembed, "W_U"):
        w_u = model.unembed.W_U
        b_u = getattr(model.unembed, "b_U", None)
        return w_u, b_u
    raise AttributeError("Could not locate the model's unembedding weights.")


def reveal_tokens_for_concept(
    model: HookedTransformer,
    x_mean: torch.Tensor,
    concept_result: Dict[str, object],
    top_k_tokens: int,
    device: str,
) -> Dict[str, object]:
    final_norm = get_final_norm_module(model)
    w_u, b_u = get_unembedding_weights(model)

    x_mean_device = x_mean.to(device)
    c_k = concept_result["concept_vector"].to(device)
    alpha = concept_result["alpha"]
    x_concept = x_mean_device + alpha * c_k

    with torch.no_grad():
        v_norm_concept = final_norm(x_concept)
        v_norm_base = final_norm(x_mean_device)
        target_device = w_u.device
        v_norm_concept = v_norm_concept.to(target_device)
        v_norm_base = v_norm_base.to(target_device)

        logits_concept = v_norm_concept @ w_u
        logits_base = v_norm_base @ w_u
        if b_u is not None:
            b_u = b_u.to(target_device)
            logits_concept = logits_concept + b_u
            logits_base = logits_base + b_u

        logits_delta = logits_concept - logits_base

        top = torch.topk(logits_delta[0], k=top_k_tokens)
        bottom = torch.topk(logits_delta[0], k=top_k_tokens, largest=False)

    top_tokens = model.tokenizer.convert_ids_to_tokens(top.indices.tolist())
    bottom_tokens = model.tokenizer.convert_ids_to_tokens(bottom.indices.tolist())

    return {
        "top_token_ids": top.indices.tolist(),
        "top_tokens": top_tokens,
        "top_token_deltas": top.values.detach().cpu().tolist(),
        "bottom_token_ids": bottom.indices.tolist(),
        "bottom_tokens": bottom_tokens,
        "bottom_token_deltas": bottom.values.detach().cpu().tolist(),
    }


def analyze_direct_pathway(
    model: HookedTransformer,
    concept_vector: torch.Tensor,
    config: PipelineConfig,
) -> Dict[str, object]:
    target_block = model.blocks[config.target_layer]
    source_block = model.blocks[config.source_layer]

    # FIX: Detach and move weights to the same device immediately
    w_in = target_block.mlp.W_in.detach().to(config.device)
    w_out = source_block.mlp.W_out.detach().to(config.device)

    # 1. Robust shape handling for Target Read Weights
    if w_in.shape[1] > w_in.shape[0] and config.target_neuron < w_in.shape[1]:
        read_weights = w_in[:, config.target_neuron]
        d_model = w_in.shape[0]
    elif config.target_neuron < w_in.shape[0]:
        read_weights = w_in[config.target_neuron, :]
        d_model = w_in.shape[1]
    else:
        raise ValueError(f"Target neuron {config.target_neuron} out of bounds for W_in shape {w_in.shape}")

    # 2. Robust RMSNorm folding
    ln_gain = 1.0  
    if hasattr(target_block, "ln2") and target_block.ln2 is not None:
        if hasattr(target_block.ln2, "w") and target_block.ln2.w is not None:
            # FIX: Also ensure layer norm gain is on the same device
            ln_gain = target_block.ln2.w.detach().to(config.device)
        elif hasattr(target_block.ln2, "weight") and target_block.ln2.weight is not None:
            ln_gain = target_block.ln2.weight.detach().to(config.device)
            
    read_weights_folded = read_weights * ln_gain # Shape: [d_model]

    # 3. Robust shape handling for Source Write Weights
    if w_out.shape[1] == d_model:
        w_out_aligned = w_out 
    elif w_out.shape[0] == d_model:
        w_out_aligned = w_out.T
    else:
        raise RuntimeError(f"W_out shape {w_out.shape} does not match extracted d_model={d_model}")

    # 4. Calculate Pathway Scores (Source Neuron Contributions)
    pathway_scores = w_out_aligned @ read_weights_folded

    top_source_neuron = int(torch.argmax(torch.abs(pathway_scores)).item())
    top_source_score = float(pathway_scores[top_source_neuron].item())

    # 5. Construct the Residual Superhighway
    residual_path_direction = pathway_scores.unsqueeze(0) @ w_out_aligned
    residual_path_direction = F.normalize(residual_path_direction, dim=1)

    # 6. Evaluate Alignment
    concept_norm = F.normalize(concept_vector.to(config.device), dim=1)
    if concept_norm.ndim == 1:
        concept_norm = concept_norm.unsqueeze(0)

    alignment = F.cosine_similarity(
        concept_norm,
        residual_path_direction,
        dim=1,
    ).item()

    return {
        "top_source_neuron": top_source_neuron,
        "top_source_score": top_source_score,
        "pathway_alignment_cosine": alignment,
        "pathway_scores_norm": float(pathway_scores.norm().item()),
        "residual_path_direction": residual_path_direction.detach().cpu(),
    }


def maybe_plot_results(
    config: PipelineConfig,
    run_records: List[Dict[str, object]],
    concept_reports: Dict[str, List[Dict[str, object]]],
) -> None:
    if not config.plot_outputs or plt is None:
        return

    run_indices = list(range(1, len(run_records) + 1))
    final_acts = [record["final_activation"] for record in run_records]
    plt.figure(figsize=(8, 4))
    plt.plot(run_indices, final_acts, marker="o")
    plt.xlabel("Run")
    plt.ylabel("Final Activation")
    plt.title("Langevin Final Activations")
    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, "langevin_final_activations.png"))
    plt.close()

    for method_name, reports in concept_reports.items():
        if not reports:
            continue
        labels = [f"{method_name}_{r['concept_index']}" for r in reports]
        drops = [r["drop_ratio"] for r in reports]
        plt.figure(figsize=(10, 4))
        plt.bar(labels, drops)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Drop Ratio")
        plt.title(f"Causal Drop Ratios ({method_name})")
        plt.tight_layout()
        plt.savefig(os.path.join(config.output_dir, f"{method_name}_drop_ratios.png"))
        plt.close()


def serialize_report(report: Dict[str, object]) -> Dict[str, object]:
    serializable: Dict[str, object] = {}
    for key, value in report.items():
        if torch.is_tensor(value):
            serializable[key] = value.tolist()
        elif isinstance(value, dict):
            serializable[key] = serialize_report(value)
        elif isinstance(value, list):
            serializable[key] = [
                serialize_report(item) if isinstance(item, dict) else item for item in value
            ]
        else:
            serializable[key] = value
    return serializable


def run_sae_corroboration(
    concept_vector: torch.Tensor,
    optimized_vector: torch.Tensor,
    sae_weights: Optional[Dict[str, torch.Tensor]],
    config: PipelineConfig,
) -> Dict[str, object]:
    if not config.sae_enabled or sae_weights is None:
        return {"status": "skipped"}

    device = config.device
    x_opt = optimized_vector.detach().to(device)
    c_k = concept_vector.detach().to(device)
    
    if x_opt.dim() == 1:
        x_opt = x_opt.unsqueeze(0)

    W_enc = sae_weights["W_enc"].to(device)
    W_dec = sae_weights["W_dec"].to(device)
    b_enc = sae_weights["b_enc"].to(device)
    b_dec = sae_weights["b_dec"].to(device)

    x_centered = x_opt - b_dec
    f_x = F.relu(x_centered @ W_enc + b_enc)

    top_activation, top_feature_idx = torch.max(f_x, dim=1)
    top_idx = int(top_feature_idx.item())
    activation_val = float(top_activation.item())

    d_sae = W_dec[top_idx, :]
    c_k_norm = F.normalize(c_k.squeeze(), dim=0)
    d_sae_norm = F.normalize(d_sae.squeeze(), dim=0)

    alignment = F.cosine_similarity(c_k_norm, d_sae_norm, dim=0).item()

    if alignment >= 0.7:
        verdict = "perfect_corroboration"
    elif alignment <= 0.3:
        verdict = "dark_feature_or_split"
    else:
        verdict = "partial_alignment"

    neuropedia_url = f"https://www.neuropedia.ai/{config.sae_neuropedia_model}/{config.sae_neuropedia_id}/{top_idx}"

    return {
        "status": "success",
        "top_sae_feature_index": top_idx,
        "top_sae_feature_activation": activation_val,
        "decoder_alignment_cosine": alignment,
        "verdict": verdict,
        "neuropedia_link": neuropedia_url,
    }


def main() -> None:
    config = PipelineConfig()
    setup_logging(config.output_dir)
    set_seed(config.seed)

    logging.info("Starting mechanistic concept pipeline on device=%s", config.device)
    logging.info("Config: %s", json.dumps(asdict(config), indent=2))

    source_hook_name = get_source_hook_name(config)
    target_hook_name = get_target_hook_name(config)

    model = HookedTransformer.from_pretrained(
        config.model_name,
        n_devices=2,                           # <-- Tells TransformerLens to use both GPUs
        dtype=get_torch_dtype(config.dtype),   # Now bfloat16
        center_unembed=False  
    )
    model.eval()
    ensure_tokenizer_padding(model)

    dummy_tokens = build_fixed_dummy_tokens(model, config.seq_len, config.device)

    logging.info("Loading dataset %s (%s)", config.dataset_name, config.dataset_split)
    dataset = load_dataset(config.dataset_name, split=config.dataset_split)

    logging.info("Phase 1: collecting empirical residual bank")
    x_bank_cpu, x_mean_cpu = collect_empirical_bank(
        model=model,
        dataset=dataset,
        config=config,
        source_hook_name=source_hook_name,
    )
    torch.save(x_bank_cpu, os.path.join(config.output_dir, "x_bank.pt"))
    torch.save(x_mean_cpu, os.path.join(config.output_dir, "x_mean.pt"))

    x_bank = x_bank_cpu.to(config.device)
    x_mean = x_mean_cpu.to(config.device)

    baseline_activation = run_activation_with_injection(
        model=model,
        dummy_tokens=dummy_tokens,
        source_hook_name=source_hook_name,
        target_hook_name=target_hook_name,
        seq_pos=config.seq_pos,
        target_neuron=config.target_neuron,
        injected_vector=x_mean,
    )
    logging.info("Baseline activation at x_mean: %.4f", baseline_activation.item())

    logging.info("Phase 2/3: multi-seed Langevin optimization")
    run_records: List[Dict[str, object]] = []
    for run_idx in range(config.num_runs):
        init_idx = torch.randint(0, x_bank.shape[0], (1,), device=x_bank.device).item()
        init_vector = x_bank[init_idx : init_idx + 1]
        record = langevin_optimize(
            model=model,
            dummy_tokens=dummy_tokens,
            x_bank=x_bank,
            initial_vector=init_vector,
            config=config,
            source_hook_name=source_hook_name,
            target_hook_name=target_hook_name,
        )
        record["init_bank_index"] = int(init_idx)
        run_records.append(record)
        logging.info(
            "Run %d/%d complete | init_idx=%d | final_activation=%.4f | kde_log_density=%.4f",
            run_idx + 1,
            config.num_runs,
            init_idx,
            record["final_activation"],
            record["kde_log_density"],
        )

    final_activations = torch.tensor([r["final_activation"] for r in run_records], dtype=torch.float32)
    max_final_activation = float(final_activations.max().item())
    convergence_threshold = config.convergence_fraction_of_best * max_final_activation
    keep_mask = final_activations >= convergence_threshold

    successful_records = [r for r, keep in zip(run_records, keep_mask.tolist()) if keep]
    if len(successful_records) < config.minimum_success_count:
        topk = min(max(config.minimum_success_count, 1), len(run_records))
        sorted_indices = torch.argsort(final_activations, descending=True)[:topk].tolist()
        successful_records = [run_records[i] for i in sorted_indices]
        logging.warning(
            "Only %d runs passed the %.2f * max activation filter. Falling back to top-%d runs.",
            int(keep_mask.sum().item()),
            config.convergence_fraction_of_best,
            topk,
        )

    successful_vectors = torch.cat([r["vector"] for r in successful_records], dim=0)
    successful_final_activations = torch.tensor(
        [r["final_activation"] for r in successful_records],
        dtype=torch.float32,
    )

    cosine_stats = compute_pairwise_cosine_stats(successful_vectors.to(config.device))
    logging.info(
        "Successful set size=%d | pairwise cosine mean=%.4f | max=%.4f",
        successful_vectors.shape[0],
        cosine_stats["mean"],
        cosine_stats["max"],
    )
    if not math.isnan(cosine_stats["max"]) and cosine_stats["max"] >= config.max_pairwise_cosine_warning:
        logging.warning(
            "Vector collapse warning: max pairwise cosine %.4f >= %.4f. Increase Langevin noise or weaken the prior.",
            cosine_stats["max"],
            config.max_pairwise_cosine_warning,
        )

    torch.save(successful_vectors, os.path.join(config.output_dir, "V_successful_rows.pt"))

    logging.info("Phase 4: decomposition with FastICA and Semi-NMF")
    decomposition_results = run_decomposition(successful_vectors=successful_vectors, config=config)

    # 1. ADD SAE LOADING LOGIC HERE (Right after decomposition_results)
    sae_weights = None
    if config.sae_enabled:
        logging.info("Downloading/Loading GemmaScope SAE via SAELens...")
        sae, _, _ = SAE.from_pretrained(
            release="gemma-scope-2b-pt-res", 
            sae_id=config.sae_path,
            device=config.device
        )
        sae_weights = {
            "W_enc": sae.W_enc.detach(), 
            "W_dec": sae.W_dec.detach(),
            "b_enc": sae.b_enc.detach(),
            "b_dec": sae.b_dec.detach(),
        }

    concept_reports: Dict[str, List[Dict[str, object]]] = {}
    token_reports: Dict[str, List[Dict[str, object]]] = {}
    pathway_reports: Dict[str, List[Dict[str, object]]] = {}
    sae_reports: Dict[str, List[Dict[str, object]]] = {}

    for method_name, decomp_result in decomposition_results.items():
        logging.info("Evaluating concepts for method=%s", method_name)
        reports = evaluate_concepts(
            model=model,
            dummy_tokens=dummy_tokens,
            x_bank=x_bank_cpu,
            x_mean=x_mean_cpu,
            successful_vectors=successful_vectors,
            final_activations=successful_final_activations,
            decomp_result=decomp_result,
            config=config,
            source_hook_name=source_hook_name,
            target_hook_name=target_hook_name,
        )
        concept_reports[method_name] = reports

        token_reports[method_name] = []
        pathway_reports[method_name] = []
        sae_reports[method_name] = []

        for report in reports:
            token_report = reveal_tokens_for_concept(
                model=model,
                x_mean=x_mean_cpu,
                concept_result=report,
                top_k_tokens=config.top_k_tokens,
                device=config.device,
            )
            token_reports[method_name].append(token_report)

            pathway_report = analyze_direct_pathway(
                model=model,
                concept_vector=report["concept_vector"],
                config=config,
            )
            pathway_reports[method_name].append(pathway_report)

            # 2. REPLACE PLACEHOLDER WITH REAL SAE FUNCTION AND PASS WEIGHTS
            sae_report = run_sae_corroboration(
                concept_vector=report["concept_vector"],
                optimized_vector=report["x_opt"],
                sae_weights=sae_weights,
                config=config,
            )
            sae_reports[method_name].append(sae_report)

            logging.info(
                "%s concept %d | drop_ratio=%.4f | cosine_precheck=%.4f | pathway_alignment=%.4f",
                method_name,
                report["concept_index"],
                report["drop_ratio"],
                report["cosine_precheck"],
                pathway_report["pathway_alignment_cosine"],
            )
            
            # 3. ADD SAE LOGGING VERDICTS HERE
            if sae_report["status"] == "success":
                logging.info(
                    "%s Concept %d | SAE Feature %d | Cosine=%.4f | Verdict: %s",
                    method_name,
                    report["concept_index"],
                    sae_report["top_sae_feature_index"],
                    sae_report["decoder_alignment_cosine"],
                    sae_report["verdict"]
                )
                logging.info(
                    "--> Verify on Neuropedia: %s", 
                    sae_report["neuropedia_link"]
                )

            logging.info(
                "%s concept %d top tokens: %s",
                method_name,
                report["concept_index"],
                token_report["top_tokens"],
            )
            logging.info(
                "%s concept %d bottom tokens: %s",
                method_name,
                report["concept_index"],
                token_report["bottom_tokens"],
            )

    maybe_plot_results(config=config, run_records=run_records, concept_reports=concept_reports)

    full_report = {
        "config": asdict(config),
        "baseline_activation": float(baseline_activation.item()),
        "max_final_activation": max_final_activation,
        "convergence_threshold": convergence_threshold,
        "pairwise_cosine_stats": cosine_stats,
        "run_records": run_records,
        "concept_reports": concept_reports,
        "token_reports": token_reports,
        "pathway_reports": pathway_reports,
        "sae_reports": sae_reports,
    }
    serialized = serialize_report(full_report)
    with open(os.path.join(config.output_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2)

    logging.info("Pipeline complete. Outputs saved to %s", config.output_dir)

if __name__ == "__main__":
    main()
