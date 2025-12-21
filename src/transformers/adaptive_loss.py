# Copyright 2024 Community Enhanced Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Adaptive Loss Utilities for Transformers Forge.

Модуль для адаптивного взвешивания loss на уровне токенов.
Позволяет улучшить качество обучения за счёт:
- Фокусировки на сложных токенах с высоким loss
- Понижения влияния простых (хорошо выученных) токенов
- Маскирования специальных токенов (pad, eos)

Key features:
- Token-level loss weighting
- Difficulty-based sampling  
- Special token masking
- Focal loss for imbalanced token distributions

Usage:
    from transformers.adaptive_loss import (
        AdaptiveLossConfig,
        compute_weighted_loss,
        create_loss_mask,
        AdaptiveLossCallback,
    )
    
    # Simple weighted loss
    config = AdaptiveLossConfig(focus_on_hard=True, gamma=2.0)
    loss = compute_weighted_loss(logits, labels, config)
    
    # In training callback
    trainer.add_callback(AdaptiveLossCallback(config))

Added in v1.1.4.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from .utils import logging


logger = logging.get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class AdaptiveLossConfig:
    """
    Configuration for adaptive loss computation.
    
    Args:
        enabled: Whether adaptive loss is enabled
        focus_on_hard: Increase weight of high-loss (hard) tokens
        gamma: Focal loss gamma (higher = more focus on hard examples)
        alpha: Base weight for loss scaling
        min_weight: Minimum weight for any token
        max_weight: Maximum weight for any token
        ignore_padding: Ignore padding tokens in loss
        ignore_special_tokens: Ignore special tokens (bos, eos, etc)
        special_token_ids: List of token IDs to ignore
        warmup_steps: Steps before applying adaptive weighting
        smooth_factor: Smoothing for loss history
        
    Example:
        >>> config = AdaptiveLossConfig(
        ...     focus_on_hard=True,
        ...     gamma=2.0,
        ...     ignore_padding=True,
        ... )
    """
    enabled: bool = True
    
    # Focal loss settings
    focus_on_hard: bool = True
    gamma: float = 2.0  # Focal loss gamma
    alpha: float = 1.0  # Base scaling factor
    
    # Weight bounds
    min_weight: float = 0.1
    max_weight: float = 5.0
    
    # Token filtering
    ignore_padding: bool = True
    ignore_special_tokens: bool = True
    special_token_ids: List[int] = field(default_factory=list)
    pad_token_id: int = -100  # Default ignore index
    
    # Training dynamics
    warmup_steps: int = 100
    smooth_factor: float = 0.99
    
    # Response-only loss (for instruction tuning)
    response_only: bool = False
    response_template_ids: Optional[List[int]] = None
    
    def __post_init__(self):
        if self.gamma < 0:
            raise ValueError("gamma must be non-negative")
        if self.min_weight > self.max_weight:
            raise ValueError("min_weight must be <= max_weight")


# =============================================================================
# Core Loss Functions
# =============================================================================


def compute_weighted_loss(
    logits: "torch.Tensor",
    labels: "torch.Tensor",
    config: Optional[AdaptiveLossConfig] = None,
    reduction: str = "mean",
) -> "torch.Tensor":
    """
    Compute weighted cross-entropy loss with adaptive token weighting.
    
    Applies focal loss and optional special token masking.
    
    Args:
        logits: Model output logits [batch, seq_len, vocab]
        labels: Target labels [batch, seq_len]
        config: AdaptiveLossConfig (default config if None)
        reduction: Loss reduction method ("mean", "sum", "none")
        
    Returns:
        Weighted loss tensor
        
    Example:
        >>> logits = model(input_ids).logits
        >>> loss = compute_weighted_loss(logits, labels)
    """
    import torch
    import torch.nn.functional as F
    
    if config is None:
        config = AdaptiveLossConfig()
    
    if not config.enabled:
        # Standard cross-entropy
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=config.pad_token_id,
            reduction=reduction,
        )
    
    # Shift for causal LM
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    batch_size, seq_len, vocab_size = shift_logits.shape
    
    # Flatten for loss computation
    flat_logits = shift_logits.view(-1, vocab_size)
    flat_labels = shift_labels.view(-1)
    
    # Compute per-token loss (no reduction)
    per_token_loss = F.cross_entropy(
        flat_logits,
        flat_labels,
        ignore_index=config.pad_token_id,
        reduction="none",
    )
    
    # Create weight mask
    weights = create_token_weights(
        per_token_loss=per_token_loss,
        labels=flat_labels,
        config=config,
    )
    
    # Apply weights
    weighted_loss = per_token_loss * weights
    
    # Create valid token mask
    valid_mask = (flat_labels != config.pad_token_id).float()
    
    # Apply reduction
    if reduction == "none":
        return weighted_loss.view(batch_size, seq_len)
    elif reduction == "sum":
        return (weighted_loss * valid_mask).sum()
    else:  # mean
        valid_count = valid_mask.sum()
        if valid_count > 0:
            return (weighted_loss * valid_mask).sum() / valid_count
        else:
            return weighted_loss.sum() * 0.0  # Avoid NaN


def create_token_weights(
    per_token_loss: "torch.Tensor",
    labels: "torch.Tensor",
    config: AdaptiveLossConfig,
) -> "torch.Tensor":
    """
    Create per-token weights based on loss values and configuration.
    
    Implements focal loss weighting: weight = (1 - p)^gamma
    where p is the probability of the correct token.
    
    Args:
        per_token_loss: Per-token cross-entropy loss
        labels: Token labels
        config: AdaptiveLossConfig
        
    Returns:
        Weight tensor same shape as per_token_loss
    """
    import torch
    
    # Start with uniform weights
    weights = torch.ones_like(per_token_loss)
    
    if not config.focus_on_hard:
        return weights * config.alpha
    
    # Convert loss back to approximate probability
    # loss = -log(p) => p = exp(-loss)
    with torch.no_grad():
        probs = torch.exp(-per_token_loss.clamp(max=10))  # Clamp for stability
        
        # Focal loss: (1 - p)^gamma
        focal_weights = (1 - probs).pow(config.gamma)
        
        # Scale by alpha
        weights = config.alpha * focal_weights
        
        # Clamp to bounds
        weights = weights.clamp(min=config.min_weight, max=config.max_weight)
    
    return weights


def create_loss_mask(
    labels: "torch.Tensor",
    tokenizer: Any = None,
    config: Optional[AdaptiveLossConfig] = None,
    response_start_positions: Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    """
    Create a mask for loss computation.
    
    Masks out padding tokens and optionally special tokens.
    For instruction tuning, can mask prompt tokens (response-only loss).
    
    Args:
        labels: Token labels [batch, seq_len]
        tokenizer: Tokenizer for special token detection
        config: AdaptiveLossConfig
        response_start_positions: Start positions of response (for response-only)
        
    Returns:
        Boolean mask [batch, seq_len] where True = compute loss
        
    Example:
        >>> mask = create_loss_mask(labels, tokenizer)
        >>> loss = (per_token_loss * mask).sum() / mask.sum()
    """
    import torch
    
    if config is None:
        config = AdaptiveLossConfig()
    
    # Start with all True (compute loss for all tokens)
    mask = torch.ones_like(labels, dtype=torch.bool)
    
    # Mask padding
    if config.ignore_padding:
        mask = mask & (labels != config.pad_token_id)
    
    # Mask special tokens
    if config.ignore_special_tokens and tokenizer is not None:
        special_ids = []
        
        if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
            special_ids.append(tokenizer.pad_token_id)
        if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
            special_ids.append(tokenizer.eos_token_id)
        if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
            special_ids.append(tokenizer.bos_token_id)
        
        special_ids.extend(config.special_token_ids)
        
        for token_id in special_ids:
            mask = mask & (labels != token_id)
    
    # Response-only masking
    if config.response_only and response_start_positions is not None:
        batch_size, seq_len = labels.shape
        positions = torch.arange(seq_len, device=labels.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        
        response_starts = response_start_positions.unsqueeze(1)
        mask = mask & (positions >= response_starts)
    
    return mask


# =============================================================================
# Focal Loss Implementation
# =============================================================================


def focal_loss(
    logits: "torch.Tensor",
    labels: "torch.Tensor",
    gamma: float = 2.0,
    alpha: float = 1.0,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> "torch.Tensor":
    """
    Compute focal loss for sequence classification.
    
    Focal loss focuses training on hard examples by down-weighting
    well-classified examples: FL = -alpha * (1 - p)^gamma * log(p)
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection"
    
    Args:
        logits: Model predictions [batch, seq_len, vocab]
        labels: Target labels [batch, seq_len]
        gamma: Focusing parameter (higher = more focus on hard examples)
        alpha: Class weight (1.0 = no class balancing)
        ignore_index: Label to ignore in loss computation
        reduction: Reduction method ("mean", "sum", "none")
        
    Returns:
        Focal loss value
        
    Example:
        >>> loss = focal_loss(logits, labels, gamma=2.0)
    """
    import torch
    import torch.nn.functional as F
    
    # Shift for causal modeling
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Flatten
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    
    # Valid mask
    valid_mask = (flat_labels != ignore_index)
    
    if not valid_mask.any():
        return logits.sum() * 0.0
    
    # Get probabilities
    log_probs = F.log_softmax(flat_logits, dim=-1)
    probs = torch.exp(log_probs)
    
    # Gather correct class probabilities
    labels_for_gather = flat_labels.clone()
    labels_for_gather[~valid_mask] = 0  # Temporary fix for gather
    
    target_log_probs = log_probs.gather(1, labels_for_gather.unsqueeze(1)).squeeze(1)
    target_probs = probs.gather(1, labels_for_gather.unsqueeze(1)).squeeze(1)
    
    # Focal loss: -alpha * (1 - p)^gamma * log(p)
    focal_weight = alpha * (1 - target_probs).pow(gamma)
    focal_loss_values = -focal_weight * target_log_probs
    
    # Apply valid mask
    focal_loss_values = focal_loss_values * valid_mask.float()
    
    if reduction == "none":
        return focal_loss_values.view(shift_labels.shape)
    elif reduction == "sum":
        return focal_loss_values.sum()
    else:  # mean
        return focal_loss_values.sum() / valid_mask.sum().float()


# =============================================================================
# Response-Only Loss for Instruction Tuning
# =============================================================================


def compute_response_only_loss(
    logits: "torch.Tensor",
    labels: "torch.Tensor",
    response_template: str,
    tokenizer: Any,
    reduction: str = "mean",
) -> "torch.Tensor":
    """
    Compute loss only on response tokens (ignore prompt).
    
    Useful for instruction tuning where you don't want the model
    to learn to generate the prompt.
    
    Args:
        logits: Model logits [batch, seq_len, vocab]
        labels: Target labels [batch, seq_len]
        response_template: String that marks start of response
                          e.g., "### Response:" or "<|assistant|>"
        tokenizer: Tokenizer for template detection
        reduction: Reduction method
        
    Returns:
        Loss computed only on response tokens
        
    Example:
        >>> loss = compute_response_only_loss(
        ...     logits, labels, 
        ...     response_template="### Answer:",
        ...     tokenizer=tokenizer
        ... )
    """
    import torch
    import torch.nn.functional as F
    
    # Get template token IDs
    template_ids = tokenizer.encode(response_template, add_special_tokens=False)
    
    # Find response start positions
    batch_size, seq_len = labels.shape
    response_starts = torch.zeros(batch_size, dtype=torch.long, device=labels.device)
    
    template_len = len(template_ids)
    
    for b in range(batch_size):
        for i in range(seq_len - template_len):
            if labels[b, i:i+template_len].tolist() == template_ids:
                response_starts[b] = i + template_len
                break
    
    # Create mask
    positions = torch.arange(seq_len, device=labels.device).unsqueeze(0)
    response_mask = (positions >= response_starts.unsqueeze(1))
    
    # Also mask padding
    pad_mask = (labels != -100) & (labels != tokenizer.pad_token_id)
    final_mask = response_mask & pad_mask
    
    # Compute loss
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = final_mask[..., 1:].contiguous()
    
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    flat_mask = shift_mask.view(-1).float()
    
    per_token_loss = F.cross_entropy(
        flat_logits, flat_labels,
        ignore_index=-100,
        reduction="none",
    )
    
    masked_loss = per_token_loss * flat_mask
    
    if reduction == "none":
        return masked_loss.view(shift_labels.shape)
    elif reduction == "sum":
        return masked_loss.sum()
    else:
        valid_tokens = flat_mask.sum()
        return masked_loss.sum() / (valid_tokens + 1e-8)


# =============================================================================
# Training Callback
# =============================================================================


class AdaptiveLossCallback:
    """
    Training callback that applies adaptive loss weighting.
    
    Can be used with HuggingFace Trainer to automatically apply
    adaptive loss during training.
    
    Note: Requires modification to Trainer or use with custom training loop.
    
    Example:
        >>> config = AdaptiveLossConfig(gamma=2.0)
        >>> callback = AdaptiveLossCallback(config)
        >>> 
        >>> # In custom training loop:
        >>> loss = callback.compute_loss(model_output.logits, batch['labels'])
    """
    
    def __init__(self, config: Optional[AdaptiveLossConfig] = None):
        """
        Initialize callback.
        
        Args:
            config: AdaptiveLossConfig or None for defaults
        """
        self.config = config or AdaptiveLossConfig()
        self.step_count = 0
        self.loss_history = []
    
    def compute_loss(
        self,
        logits: "torch.Tensor",
        labels: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Compute adaptive loss.
        
        Args:
            logits: Model logits
            labels: Target labels
            
        Returns:
            Weighted loss
        """
        # Check warmup
        if self.step_count < self.config.warmup_steps:
            # During warmup, use standard loss
            temp_config = AdaptiveLossConfig(enabled=False)
            loss = compute_weighted_loss(logits, labels, temp_config)
        else:
            loss = compute_weighted_loss(logits, labels, self.config)
        
        self.step_count += 1
        self.loss_history.append(loss.item())
        
        return loss
    
    def get_stats(self) -> Dict[str, float]:
        """Get training statistics."""
        if not self.loss_history:
            return {}
        
        recent = self.loss_history[-100:]
        return {
            "adaptive_loss/mean": sum(recent) / len(recent),
            "adaptive_loss/min": min(recent),
            "adaptive_loss/max": max(recent),
            "adaptive_loss/steps": self.step_count,
        }
    
    def reset(self):
        """Reset callback state."""
        self.step_count = 0
        self.loss_history = []


# =============================================================================
# Utility Functions
# =============================================================================


def get_token_loss_distribution(
    logits: "torch.Tensor",
    labels: "torch.Tensor",
    tokenizer: Any = None,
    top_k: int = 20,
) -> Dict[str, Any]:
    """
    Analyze per-token loss distribution.
    
    Useful for understanding which tokens are hardest to predict.
    
    Args:
        logits: Model logits [batch, seq_len, vocab]
        labels: Target labels [batch, seq_len]  
        tokenizer: Optional tokenizer for token decoding
        top_k: Number of top loss tokens to return
        
    Returns:
        Dict with loss statistics and hardest tokens
        
    Example:
        >>> stats = get_token_loss_distribution(logits, labels, tokenizer)
        >>> print(f"Hardest tokens: {stats['hardest_tokens']}")
    """
    import torch
    import torch.nn.functional as F
    
    # Shift
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Per-token loss
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    
    per_token_loss = F.cross_entropy(
        flat_logits, flat_labels,
        ignore_index=-100,
        reduction="none",
    )
    
    # Valid mask
    valid_mask = (flat_labels != -100)
    valid_losses = per_token_loss[valid_mask]
    valid_labels = flat_labels[valid_mask]
    
    # Statistics
    stats = {
        "mean_loss": valid_losses.mean().item(),
        "std_loss": valid_losses.std().item(),
        "min_loss": valid_losses.min().item(),
        "max_loss": valid_losses.max().item(),
        "median_loss": valid_losses.median().item(),
    }
    
    # Find hardest tokens
    if top_k > 0:
        top_losses, top_indices = valid_losses.topk(min(top_k, len(valid_losses)))
        top_tokens = valid_labels[top_indices]
        
        hardest = []
        for i, (loss, token_id) in enumerate(zip(top_losses, top_tokens)):
            entry = {
                "rank": i + 1,
                "loss": loss.item(),
                "token_id": token_id.item(),
            }
            if tokenizer is not None:
                entry["token"] = tokenizer.decode([token_id.item()])
            hardest.append(entry)
        
        stats["hardest_tokens"] = hardest
    
    return stats
