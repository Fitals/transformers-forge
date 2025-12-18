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
Layer Utilities for Transformers Enhanced.

This module provides safe utilities for layer freezing and analysis.
All functions work with `requires_grad` and do NOT modify model 
architecture or weights.

Key benefits:
- Memory savings (frozen layers don't store gradients)
- Faster training (fewer parameters to update)
- LP-LoRA style training (freeze early layers)
- Gradual unfreezing for better transfer learning

Usage:
    from transformers.layer_utils import (
        freeze_first_n_layers,
        freeze_embeddings,
        get_trainable_params,
        print_layer_status
    )
    
    # Freeze first 16 layers (LP-LoRA style)
    freeze_first_n_layers(model, n=16)
    
    # Check status
    print_layer_status(model)
"""

import re
from typing import Any, Callable, Dict, List, Optional, Pattern, Set, Tuple, Union

from .utils import logging


logger = logging.get_logger(__name__)


# =============================================================================
# Core Freezing Functions
# =============================================================================


def freeze_model(model: Any) -> int:
    """
    Freeze all parameters in the model.
    
    Args:
        model: PyTorch model (nn.Module)
        
    Returns:
        Number of parameters frozen
        
    Example:
        >>> freeze_model(model)
        >>> # All parameters now have requires_grad=False
    """
    frozen_count = 0
    for param in model.parameters():
        if param.requires_grad:
            param.requires_grad = False
            frozen_count += param.numel()
    
    logger.info(f"Frozen {frozen_count:,} parameters")
    return frozen_count


def unfreeze_model(model: Any) -> int:
    """
    Unfreeze all parameters in the model.
    
    Args:
        model: PyTorch model (nn.Module)
        
    Returns:
        Number of parameters unfrozen
        
    Example:
        >>> unfreeze_model(model)
        >>> # All parameters now have requires_grad=True
    """
    unfrozen_count = 0
    for param in model.parameters():
        if not param.requires_grad:
            param.requires_grad = True
            unfrozen_count += param.numel()
    
    logger.info(f"Unfrozen {unfrozen_count:,} parameters")
    return unfrozen_count


def freeze_by_name(
    model: Any,
    patterns: Union[str, List[str]],
    case_sensitive: bool = True
) -> int:
    """
    Freeze parameters matching name patterns.
    
    Args:
        model: PyTorch model
        patterns: String or list of regex patterns to match parameter names
        case_sensitive: Whether pattern matching is case-sensitive
        
    Returns:
        Number of parameters frozen
        
    Example:
        >>> # Freeze all attention layers
        >>> freeze_by_name(model, [".*attention.*", ".*attn.*"])
        
        >>> # Freeze specific layer
        >>> freeze_by_name(model, "layers.0.")
    """
    if isinstance(patterns, str):
        patterns = [patterns]
    
    flags = 0 if case_sensitive else re.IGNORECASE
    compiled_patterns = [re.compile(p, flags) for p in patterns]
    
    frozen_count = 0
    for name, param in model.named_parameters():
        if any(p.search(name) for p in compiled_patterns):
            if param.requires_grad:
                param.requires_grad = False
                frozen_count += param.numel()
    
    logger.info(f"Frozen {frozen_count:,} parameters matching patterns")
    return frozen_count


def unfreeze_by_name(
    model: Any,
    patterns: Union[str, List[str]],
    case_sensitive: bool = True
) -> int:
    """
    Unfreeze parameters matching name patterns.
    
    Args:
        model: PyTorch model
        patterns: String or list of regex patterns to match parameter names
        case_sensitive: Whether pattern matching is case-sensitive
        
    Returns:
        Number of parameters unfrozen
        
    Example:
        >>> # Unfreeze all attention layers
        >>> unfreeze_by_name(model, [".*attention.*"])
    """
    if isinstance(patterns, str):
        patterns = [patterns]
    
    flags = 0 if case_sensitive else re.IGNORECASE
    compiled_patterns = [re.compile(p, flags) for p in patterns]
    
    unfrozen_count = 0
    for name, param in model.named_parameters():
        if any(p.search(name) for p in compiled_patterns):
            if not param.requires_grad:
                param.requires_grad = True
                unfrozen_count += param.numel()
    
    logger.info(f"Unfrozen {unfrozen_count:,} parameters matching patterns")
    return unfrozen_count


# =============================================================================
# Smart Freezing for Transformer Models
# =============================================================================


def _detect_layer_pattern(model: Any) -> Tuple[str, int]:
    """
    Auto-detect layer naming pattern and count.
    
    Common patterns:
    - model.layers.{N}       (Llama, Qwen, Mistral)
    - model.transformer.h.{N} (GPT-2, GPT-J)
    - model.encoder.layer.{N} (BERT)
    - model.decoder.layers.{N} (T5 decoder)
    
    Returns:
        Tuple of (pattern_prefix, num_layers)
    """
    layer_patterns = [
        (r"model\.layers\.(\d+)\.", "model.layers."),
        (r"transformer\.h\.(\d+)\.", "transformer.h."),
        (r"model\.transformer\.h\.(\d+)\.", "model.transformer.h."),
        (r"encoder\.layer\.(\d+)\.", "encoder.layer."),
        (r"decoder\.layers\.(\d+)\.", "decoder.layers."),
        (r"model\.decoder\.layers\.(\d+)\.", "model.decoder.layers."),
        (r"layers\.(\d+)\.", "layers."),
        (r"h\.(\d+)\.", "h."),
    ]
    
    layer_indices: Set[int] = set()
    detected_pattern = None
    
    for name, _ in model.named_parameters():
        for regex, prefix in layer_patterns:
            match = re.search(regex, name)
            if match:
                layer_indices.add(int(match.group(1)))
                detected_pattern = prefix
                break
        if detected_pattern:
            break
    
    if detected_pattern is None:
        return "", 0
    
    # Find all layers with this pattern
    for name, _ in model.named_parameters():
        match = re.search(rf"{re.escape(detected_pattern)}(\d+)\.", name)
        if match:
            layer_indices.add(int(match.group(1)))
    
    num_layers = len(layer_indices)
    logger.debug(f"Detected pattern: '{detected_pattern}' with {num_layers} layers")
    
    return detected_pattern, num_layers


def get_num_layers(model: Any) -> int:
    """
    Get the number of transformer layers in the model.
    
    Args:
        model: PyTorch transformer model
        
    Returns:
        Number of layers detected
        
    Example:
        >>> num = get_num_layers(model)
        >>> print(f"Model has {num} layers")
    """
    _, num_layers = _detect_layer_pattern(model)
    return num_layers


def freeze_first_n_layers(model: Any, n: int) -> int:
    """
    Freeze the first N transformer layers.
    
    This is the LP-LoRA technique - freezing early layers while
    training later layers for better efficiency.
    
    Args:
        model: PyTorch transformer model
        n: Number of layers to freeze (from the beginning)
        
    Returns:
        Number of parameters frozen
        
    Example:
        >>> # For a 32-layer model, freeze first 16
        >>> freeze_first_n_layers(model, 16)
        >>> # Now only last 16 layers are trainable
    """
    pattern, num_layers = _detect_layer_pattern(model)
    
    if not pattern:
        logger.warning("Could not detect layer pattern. Use freeze_by_name() instead.")
        return 0
    
    if n > num_layers:
        logger.warning(f"Requested freezing {n} layers but model only has {num_layers}")
        n = num_layers
    
    # Create patterns for layers 0 to n-1
    frozen_count = 0
    for layer_idx in range(n):
        layer_pattern = f"{pattern}{layer_idx}."
        for name, param in model.named_parameters():
            if layer_pattern in name and param.requires_grad:
                param.requires_grad = False
                frozen_count += param.numel()
    
    logger.info(f"Frozen first {n}/{num_layers} layers ({frozen_count:,} parameters)")
    return frozen_count


def freeze_last_n_layers(model: Any, n: int) -> int:
    """
    Freeze the last N transformer layers.
    
    Args:
        model: PyTorch transformer model
        n: Number of layers to freeze (from the end)
        
    Returns:
        Number of parameters frozen
        
    Example:
        >>> # Freeze last 8 layers
        >>> freeze_last_n_layers(model, 8)
    """
    pattern, num_layers = _detect_layer_pattern(model)
    
    if not pattern:
        logger.warning("Could not detect layer pattern. Use freeze_by_name() instead.")
        return 0
    
    if n > num_layers:
        n = num_layers
    
    frozen_count = 0
    for layer_idx in range(num_layers - n, num_layers):
        layer_pattern = f"{pattern}{layer_idx}."
        for name, param in model.named_parameters():
            if layer_pattern in name and param.requires_grad:
                param.requires_grad = False
                frozen_count += param.numel()
    
    logger.info(f"Frozen last {n}/{num_layers} layers ({frozen_count:,} parameters)")
    return frozen_count


def freeze_except_last_n(model: Any, n: int) -> int:
    """
    Freeze everything except the last N layers.
    
    Useful for efficient fine-tuning where you only train
    the final layers.
    
    Args:
        model: PyTorch transformer model
        n: Number of layers to keep trainable (from the end)
        
    Returns:
        Number of parameters frozen
        
    Example:
        >>> # Train only last 4 layers
        >>> freeze_except_last_n(model, 4)
    """
    pattern, num_layers = _detect_layer_pattern(model)
    
    if not pattern:
        logger.warning("Could not detect layer pattern.")
        return 0
    
    # Freeze all first
    frozen_count = freeze_model(model)
    
    # Unfreeze last n layers
    for layer_idx in range(num_layers - n, num_layers):
        layer_pattern = f"{pattern}{layer_idx}."
        for name, param in model.named_parameters():
            if layer_pattern in name:
                param.requires_grad = True
    
    # Also unfreeze lm_head / output projection
    unfreeze_by_name(model, [
        r"lm_head",
        r"output",
        r"cls",
        r"classifier",
    ])
    
    trainable = get_trainable_params(model)
    logger.info(f"Kept last {n} layers trainable ({trainable:,} parameters)")
    return frozen_count


def freeze_embeddings(model: Any) -> int:
    """
    Freeze embedding layers.
    
    Freezing embeddings prevents the model from "forgetting"
    its vocabulary knowledge during fine-tuning.
    
    Args:
        model: PyTorch transformer model
        
    Returns:
        Number of parameters frozen
        
    Example:
        >>> freeze_embeddings(model)
        >>> # Embeddings are now frozen
    """
    patterns = [
        r"embed",
        r"wte",
        r"wpe",
        r"word_embedding",
        r"position_embedding",
        r"token_type_embedding",
    ]
    
    return freeze_by_name(model, patterns, case_sensitive=False)


def freeze_lm_head(model: Any) -> int:
    """
    Freeze the language model head / output layer.
    
    Args:
        model: PyTorch transformer model
        
    Returns:
        Number of parameters frozen
    """
    patterns = [
        r"lm_head",
        r"output",
        r"cls\.",
        r"classifier",
    ]
    
    return freeze_by_name(model, patterns, case_sensitive=False)


# =============================================================================
# Analysis Functions
# =============================================================================


def get_trainable_params(model: Any) -> int:
    """
    Count trainable (non-frozen) parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
        
    Example:
        >>> trainable = get_trainable_params(model)
        >>> print(f"Trainable: {trainable:,}")
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_frozen_params(model: Any) -> int:
    """
    Count frozen parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of frozen parameters
    """
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)


def get_total_params(model: Any) -> int:
    """
    Count total parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def get_frozen_percentage(model: Any) -> float:
    """
    Get percentage of frozen parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Percentage of frozen parameters (0-100)
        
    Example:
        >>> pct = get_frozen_percentage(model)
        >>> print(f"Frozen: {pct:.1f}%")
    """
    total = get_total_params(model)
    if total == 0:
        return 0.0
    return (get_frozen_params(model) / total) * 100


def get_layer_status(model: Any) -> List[Dict[str, Any]]:
    """
    Get detailed status of each layer.
    
    Args:
        model: PyTorch transformer model
        
    Returns:
        List of dicts with layer info
        
    Example:
        >>> layers = get_layer_status(model)
        >>> for layer in layers:
        ...     print(f"{layer['name']}: {layer['trainable']}")
    """
    pattern, num_layers = _detect_layer_pattern(model)
    
    if not pattern:
        return []
    
    layers = []
    for layer_idx in range(num_layers):
        layer_pattern = f"{pattern}{layer_idx}."
        
        layer_params = 0
        trainable_params = 0
        
        for name, param in model.named_parameters():
            if layer_pattern in name:
                layer_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
        
        layers.append({
            "index": layer_idx,
            "name": f"{pattern}{layer_idx}",
            "total_params": layer_params,
            "trainable_params": trainable_params,
            "frozen_params": layer_params - trainable_params,
            "trainable": trainable_params > 0,
            "frozen_pct": ((layer_params - trainable_params) / layer_params * 100) if layer_params > 0 else 0,
        })
    
    return layers


def print_layer_status(model: Any, show_all: bool = False):
    """
    Print a formatted table of layer status.
    
    Args:
        model: PyTorch transformer model
        show_all: If True, show all layers; otherwise show summary
        
    Example:
        >>> print_layer_status(model)
        
        ========================================
        LAYER STATUS
        ========================================
        Layer 0-15:  FROZEN  (1.2B params)
        Layer 16-31: ACTIVE  (1.2B params)
        ----------------------------------------
        Embeddings:  FROZEN
        LM Head:     ACTIVE
        ========================================
        Total: 2.4B | Trainable: 1.2B (50.0%)
        ========================================
    """
    layers = get_layer_status(model)
    
    if not layers:
        print("Could not detect layer structure.")
        print(f"Total params: {get_total_params(model):,}")
        print(f"Trainable:    {get_trainable_params(model):,}")
        return
    
    print("\n" + "=" * 50)
    print("LAYER STATUS")
    print("=" * 50)
    
    if show_all:
        for layer in layers:
            status = "ACTIVE" if layer["trainable"] else "FROZEN"
            params = layer["total_params"]
            print(f"Layer {layer['index']:2d}: {status:6s} ({params:,} params)")
    else:
        # Group consecutive frozen/active layers
        groups = []
        current_group = {
            "start": 0,
            "end": 0,
            "trainable": layers[0]["trainable"],
            "params": layers[0]["total_params"]
        }
        
        for i, layer in enumerate(layers[1:], 1):
            if layer["trainable"] == current_group["trainable"]:
                current_group["end"] = i
                current_group["params"] += layer["total_params"]
            else:
                groups.append(current_group)
                current_group = {
                    "start": i,
                    "end": i,
                    "trainable": layer["trainable"],
                    "params": layer["total_params"]
                }
        groups.append(current_group)
        
        for group in groups:
            status = "ACTIVE" if group["trainable"] else "FROZEN"
            if group["start"] == group["end"]:
                layer_str = f"Layer {group['start']}"
            else:
                layer_str = f"Layer {group['start']}-{group['end']}"
            
            params_str = _format_params(group["params"])
            print(f"{layer_str:15s} {status:6s} ({params_str})")
    
    print("-" * 50)
    
    # Check embeddings and lm_head
    embed_frozen = all(
        not p.requires_grad 
        for n, p in model.named_parameters() 
        if "embed" in n.lower()
    )
    
    lm_head_frozen = all(
        not p.requires_grad 
        for n, p in model.named_parameters() 
        if "lm_head" in n.lower() or n.lower().endswith(".output")
    )
    
    print(f"Embeddings:     {'FROZEN' if embed_frozen else 'ACTIVE'}")
    print(f"LM Head:        {'FROZEN' if lm_head_frozen else 'ACTIVE'}")
    
    print("=" * 50)
    
    total = get_total_params(model)
    trainable = get_trainable_params(model)
    pct = (trainable / total * 100) if total > 0 else 0
    
    print(f"Total: {_format_params(total)} | Trainable: {_format_params(trainable)} ({pct:.1f}%)")
    print("=" * 50 + "\n")


def _format_params(count: int) -> str:
    """Format parameter count in human-readable form."""
    if count >= 1e9:
        return f"{count / 1e9:.2f}B"
    elif count >= 1e6:
        return f"{count / 1e6:.2f}M"
    elif count >= 1e3:
        return f"{count / 1e3:.2f}K"
    return str(count)


# =============================================================================
# Gradual Unfreezing
# =============================================================================


class GradualUnfreezer:
    """
    Implements gradual unfreezing for better transfer learning.
    
    Gradually unfreezes layers from the end of the model to the
    beginning, allowing top layers to adapt first.
    
    Based on: "Universal Language Model Fine-tuning for Text Classification"
    (Howard & Ruder, 2018)
    
    Example:
        >>> unfreezer = GradualUnfreezer(model, total_epochs=10)
        >>> 
        >>> for epoch in range(10):
        ...     unfreezer.step(epoch)
        ...     train_one_epoch(model, ...)
    """
    
    def __init__(
        self,
        model: Any,
        total_epochs: int,
        unfreeze_embeddings_at: Optional[int] = None,
        freeze_embeddings: bool = True,
        verbose: bool = True
    ):
        """
        Initialize gradual unfreezer.
        
        Args:
            model: PyTorch transformer model
            total_epochs: Total number of training epochs
            unfreeze_embeddings_at: Epoch at which to unfreeze embeddings (None = never)
            freeze_embeddings: Whether to freeze embeddings initially
            verbose: Print status messages
        """
        self.model = model
        self.total_epochs = total_epochs
        self.unfreeze_embeddings_at = unfreeze_embeddings_at
        self.freeze_embeddings_flag = freeze_embeddings
        self.verbose = verbose
        
        # Detect layer structure
        self.pattern, self.num_layers = _detect_layer_pattern(model)
        
        if self.num_layers == 0:
            logger.warning("Could not detect layers. Gradual unfreezing disabled.")
            return
        
        # Calculate unfreezing schedule
        # Unfreeze layers from end to beginning over epochs
        self.layers_per_epoch = max(1, self.num_layers // total_epochs)
        
        # Initial state: freeze all, unfreeze lm_head
        freeze_model(model)
        unfreeze_by_name(model, [r"lm_head", r"output", r"cls"])
        
        if self.verbose:
            logger.info(
                f"GradualUnfreezer initialized: {self.num_layers} layers, "
                f"~{self.layers_per_epoch} layers/epoch"
            )
    
    def step(self, epoch: int):
        """
        Unfreeze appropriate layers for the current epoch.
        
        Args:
            epoch: Current epoch (0-indexed)
        """
        if self.num_layers == 0:
            return
        
        # Calculate how many layers should be unfrozen by now
        layers_to_unfreeze = min(
            (epoch + 1) * self.layers_per_epoch,
            self.num_layers
        )
        
        # Unfreeze from the end
        for layer_idx in range(self.num_layers - layers_to_unfreeze, self.num_layers):
            layer_pattern = f"{self.pattern}{layer_idx}."
            for name, param in self.model.named_parameters():
                if layer_pattern in name:
                    param.requires_grad = True
        
        # Check if we should unfreeze embeddings
        if (self.unfreeze_embeddings_at is not None and 
            epoch >= self.unfreeze_embeddings_at):
            unfreeze_by_name(self.model, [r"embed"], case_sensitive=False)
        
        if self.verbose:
            trainable_pct = 100 - get_frozen_percentage(self.model)
            logger.info(
                f"Epoch {epoch}: Unfrozen {layers_to_unfreeze}/{self.num_layers} layers "
                f"({trainable_pct:.1f}% trainable)"
            )
    
    def get_current_trainable_layers(self) -> int:
        """Get number of currently trainable layers."""
        count = 0
        for layer in get_layer_status(self.model):
            if layer["trainable"]:
                count += 1
        return count


# =============================================================================
# Convenience Functions
# =============================================================================


def setup_lp_lora_style(
    model: Any,
    freeze_ratio: float = 0.5,
    freeze_embed: bool = True
) -> int:
    """
    Setup LP-LoRA style training: freeze early layers.
    
    LP-LoRA (Layer-wise Pre-training for LoRA) shows that freezing
    early layers and training only later layers gives good results
    with significant memory savings.
    
    Args:
        model: PyTorch transformer model
        freeze_ratio: Ratio of layers to freeze (0.0 to 1.0)
        freeze_embed: Whether to freeze embeddings
        
    Returns:
        Number of parameters frozen
        
    Example:
        >>> # Freeze 50% of layers
        >>> setup_lp_lora_style(model, freeze_ratio=0.5)
    """
    num_layers = get_num_layers(model)
    layers_to_freeze = int(num_layers * freeze_ratio)
    
    frozen = freeze_first_n_layers(model, layers_to_freeze)
    
    if freeze_embed:
        frozen += freeze_embeddings(model)
    
    return frozen


def get_memory_savings_estimate(model: Any) -> Dict[str, float]:
    """
    Estimate memory savings from current freezing configuration.
    
    Args:
        model: PyTorch transformer model
        
    Returns:
        Dict with estimated savings
        
    Example:
        >>> savings = get_memory_savings_estimate(model)
        >>> print(f"Gradient memory saved: {savings['gradient_gb']:.2f} GB")
    """
    total = get_total_params(model)
    frozen = get_frozen_params(model)
    trainable = get_trainable_params(model)
    
    # Assuming float32 gradients = 4 bytes per param
    bytes_per_param = 4
    
    frozen_gradient_bytes = frozen * bytes_per_param
    trainable_gradient_bytes = trainable * bytes_per_param
    
    # Optimizer states (Adam has 2x overhead)
    frozen_optim_bytes = frozen * bytes_per_param * 2
    trainable_optim_bytes = trainable * bytes_per_param * 2
    
    return {
        "frozen_params": frozen,
        "trainable_params": trainable,
        "frozen_percentage": (frozen / total * 100) if total > 0 else 0,
        "gradient_saved_gb": frozen_gradient_bytes / (1024**3),
        "optimizer_saved_gb": frozen_optim_bytes / (1024**3),
        "total_saved_gb": (frozen_gradient_bytes + frozen_optim_bytes) / (1024**3),
    }
