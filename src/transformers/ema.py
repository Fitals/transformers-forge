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
Exponential Moving Average (EMA) for Transformers Enhanced.

EMA maintains a smoothed version of model weights that typically 
generalizes better than the final training weights.

Key benefits:
- Better generalization (proven by Polyak averaging theory)
- More stable predictions (less noise in weights)
- Used in SOTA models (Stable Diffusion, DALL-E, etc.)

Typical improvement: +1-3% on evaluation metrics.

Usage:
    from transformers import Trainer
    from transformers.ema import EMACallback
    
    trainer = Trainer(
        model=model,
        args=args,
        callbacks=[EMACallback(decay=0.999)]
    )
    trainer.train()
    
    # Get EMA weights (better quality)
    ema_state = trainer.callback_handler.callbacks[0].get_ema_state()
"""

import copy
from typing import Any, Dict, Optional

from .trainer_callback import TrainerCallback
from .utils import logging


logger = logging.get_logger(__name__)


# =============================================================================
# EMA State Management
# =============================================================================


def create_ema_state(model: Any) -> Dict[str, Any]:
    """
    Create initial EMA state from model parameters.
    
    Args:
        model: PyTorch model (nn.Module)
        
    Returns:
        Dictionary mapping parameter names to cloned tensors
        
    Example:
        >>> ema_state = create_ema_state(model)
    """
    try:
        import torch
    except ImportError:
        raise ImportError("EMA requires PyTorch")
    
    ema_state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema_state[name] = param.data.clone().detach()
    
    return ema_state


def update_ema_state(
    model: Any,
    ema_state: Dict[str, Any],
    decay: float = 0.999
) -> None:
    """
    Update EMA state with current model parameters.
    
    Formula: ema = decay * ema + (1 - decay) * current
    
    Args:
        model: PyTorch model
        ema_state: Current EMA state dictionary
        decay: EMA decay factor (0.999 = slow update, 0.99 = faster)
    """
    try:
        import torch
    except ImportError:
        return
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in ema_state and param.requires_grad:
                # EMA update: ema = decay * ema + (1 - decay) * param
                ema_state[name].mul_(decay).add_(param.data, alpha=1 - decay)


def apply_ema_state(
    model: Any,
    ema_state: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Apply EMA state to model and return original state.
    
    Args:
        model: PyTorch model
        ema_state: EMA state to apply
        
    Returns:
        Original model state (for restoration if needed)
        
    Example:
        >>> original = apply_ema_state(model, ema_state)
        >>> # model now has EMA weights
        >>> # To restore:
        >>> apply_ema_state(model, original)
    """
    try:
        import torch
    except ImportError:
        return {}
    
    original_state = {}
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in ema_state:
                # Save original
                original_state[name] = param.data.clone()
                # Apply EMA
                param.data.copy_(ema_state[name])
    
    return original_state


def get_ema_state_dict(ema_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert EMA state to a state_dict for saving.
    
    Args:
        ema_state: EMA state dictionary
        
    Returns:
        State dict suitable for torch.save()
    """
    return {name: tensor.cpu().clone() for name, tensor in ema_state.items()}


def load_ema_state_dict(
    ema_state_dict: Dict[str, Any],
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load EMA state dict, optionally moving to device.
    
    Args:
        ema_state_dict: Saved EMA state dict
        device: Target device (e.g., 'cuda', 'cpu')
        
    Returns:
        EMA state ready for use
    """
    try:
        import torch
    except ImportError:
        return ema_state_dict
    
    if device is None:
        return {name: tensor.clone() for name, tensor in ema_state_dict.items()}
    
    return {
        name: tensor.to(device) 
        for name, tensor in ema_state_dict.items()
    }


# =============================================================================
# EMA Callback for Trainer
# =============================================================================


class EMACallback(TrainerCallback):
    """
    Trainer callback that maintains EMA weights.
    
    Exponential Moving Average (EMA) creates a smoothed version of 
    model weights that typically generalizes better than raw training 
    weights.
    
    Args:
        decay: EMA decay factor. Higher = slower updates.
            - 0.999: Update ~0.1% per step (recommended for most cases)
            - 0.9999: Update ~0.01% per step (for very long training)
            - 0.99: Update ~1% per step (faster adaptation)
        update_after_step: Start EMA after this many steps (warmup)
        update_every: Update EMA every N steps (1 = every step)
        
    Attributes:
        ema_state: Current EMA state dictionary
        
    Example:
        >>> from transformers import Trainer
        >>> from transformers.ema import EMACallback
        >>> 
        >>> ema_callback = EMACallback(decay=0.999)
        >>> trainer = Trainer(
        ...     model=model,
        ...     args=args,
        ...     callbacks=[ema_callback]
        ... )
        >>> trainer.train()
        >>> 
        >>> # Apply EMA weights for inference
        >>> ema_callback.apply_ema(model)
        >>> 
        >>> # Or get the state for saving
        >>> ema_state = ema_callback.get_ema_state()
        >>> torch.save(ema_state, "ema_weights.pt")
    """
    
    def __init__(
        self,
        decay: float = 0.999,
        update_after_step: int = 0,
        update_every: int = 1
    ):
        self.decay = decay
        self.update_after_step = update_after_step
        self.update_every = update_every
        
        self.ema_state: Optional[Dict[str, Any]] = None
        self._step_count = 0
        self._original_state: Optional[Dict[str, Any]] = None
        
        # Validate decay
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        
        logger.info(
            f"EMACallback initialized: decay={decay}, "
            f"update_after_step={update_after_step}, "
            f"update_every={update_every}"
        )
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Initialize EMA state at the start of training."""
        if model is None:
            logger.warning("EMACallback: No model provided, EMA disabled")
            return
        
        self.ema_state = create_ema_state(model)
        self._step_count = 0
        
        # Calculate memory usage
        total_params = sum(p.numel() for p in self.ema_state.values())
        memory_mb = total_params * 4 / (1024 * 1024)  # float32
        
        logger.info(
            f"EMA initialized: {total_params:,} parameters, "
            f"~{memory_mb:.1f} MB additional memory"
        )
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Update EMA state after each training step."""
        if self.ema_state is None or model is None:
            return
        
        self._step_count += 1
        
        # Skip warmup period
        if self._step_count < self.update_after_step:
            return
        
        # Update only every N steps
        if self._step_count % self.update_every != 0:
            return
        
        update_ema_state(model, self.ema_state, self.decay)
    
    def on_train_end(self, args, state, control, model=None, **kwargs):
        """Log EMA statistics at end of training."""
        if self.ema_state is None:
            return
        
        logger.info(
            f"EMA training complete: {self._step_count} steps, "
            f"decay={self.decay}"
        )
    
    def apply_ema(self, model: Any) -> Dict[str, Any]:
        """
        Apply EMA weights to model.
        
        Args:
            model: PyTorch model
            
        Returns:
            Original state (for restoration)
            
        Example:
            >>> original = ema_callback.apply_ema(model)
            >>> # model now has EMA weights
            >>> outputs = model.generate(...)
            >>> 
            >>> # Restore original weights
            >>> ema_callback.restore_original(model, original)
        """
        if self.ema_state is None:
            raise RuntimeError("EMA not initialized. Train the model first.")
        
        self._original_state = apply_ema_state(model, self.ema_state)
        logger.info("EMA weights applied to model")
        return self._original_state
    
    def restore_original(
        self, 
        model: Any, 
        original_state: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Restore original (non-EMA) weights.
        
        Args:
            model: PyTorch model
            original_state: Original state to restore (optional, uses cached)
        """
        state_to_restore = original_state or self._original_state
        
        if state_to_restore is None:
            raise RuntimeError("No original state to restore")
        
        apply_ema_state(model, state_to_restore)
        logger.info("Original weights restored")
    
    def get_ema_state(self) -> Dict[str, Any]:
        """
        Get current EMA state for saving.
        
        Returns:
            EMA state dictionary (CPU tensors)
        """
        if self.ema_state is None:
            raise RuntimeError("EMA not initialized")
        
        return get_ema_state_dict(self.ema_state)
    
    def load_ema_state(
        self, 
        ema_state_dict: Dict[str, Any],
        device: Optional[str] = None
    ) -> None:
        """
        Load EMA state from saved dictionary.
        
        Args:
            ema_state_dict: Saved EMA state
            device: Target device
        """
        self.ema_state = load_ema_state_dict(ema_state_dict, device)
        logger.info("EMA state loaded")
    
    def get_decay(self) -> float:
        """Get current decay value."""
        return self.decay
    
    def set_decay(self, decay: float) -> None:
        """
        Dynamically update decay value.
        
        Useful for decay scheduling.
        
        Args:
            decay: New decay value
        """
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        
        self.decay = decay
        logger.info(f"EMA decay updated to {decay}")


# =============================================================================
# EMA Model Wrapper
# =============================================================================


class EMAModel:
    """
    Wrapper that manages both original and EMA versions of a model.
    
    This provides a convenient interface for models that need to
    switch between training weights and EMA weights.
    
    Args:
        model: PyTorch model to wrap
        decay: EMA decay factor
        
    Example:
        >>> ema_model = EMAModel(model, decay=0.999)
        >>> 
        >>> # During training:
        >>> for batch in dataloader:
        ...     outputs = ema_model.model(batch)  # Use training weights
        ...     loss.backward()
        ...     optimizer.step()
        ...     ema_model.update()  # Update EMA
        >>> 
        >>> # For inference:
        >>> with ema_model.use_ema():
        ...     outputs = ema_model.model(batch)  # Uses EMA weights
    """
    
    def __init__(self, model: Any, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.ema_state = create_ema_state(model)
        self._in_ema_context = False
        self._original_state: Optional[Dict[str, Any]] = None
    
    def update(self) -> None:
        """Update EMA state with current model weights."""
        if self._in_ema_context:
            raise RuntimeError("Cannot update EMA while using EMA weights")
        
        update_ema_state(self.model, self.ema_state, self.decay)
    
    def use_ema(self):
        """
        Context manager to temporarily use EMA weights.
        
        Example:
            >>> with ema_model.use_ema():
            ...     outputs = ema_model.model(inputs)
        """
        return _EMAContext(self)
    
    def apply_ema(self) -> None:
        """Permanently apply EMA weights to model."""
        apply_ema_state(self.model, self.ema_state)
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get EMA state dict for saving."""
        return get_ema_state_dict(self.ema_state)
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load EMA state dict."""
        self.ema_state = load_ema_state_dict(state_dict)


class _EMAContext:
    """Context manager for temporarily using EMA weights."""
    
    def __init__(self, ema_model: EMAModel):
        self.ema_model = ema_model
    
    def __enter__(self):
        if self.ema_model._in_ema_context:
            raise RuntimeError("Already in EMA context")
        
        self.ema_model._in_ema_context = True
        self.ema_model._original_state = apply_ema_state(
            self.ema_model.model, 
            self.ema_model.ema_state
        )
        return self.ema_model.model
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        apply_ema_state(
            self.ema_model.model, 
            self.ema_model._original_state
        )
        self.ema_model._in_ema_context = False
        self.ema_model._original_state = None
        return False


# =============================================================================
# Utility Functions
# =============================================================================


def compute_optimal_decay(
    total_steps: int,
    target_half_life_steps: Optional[int] = None
) -> float:
    """
    Compute optimal EMA decay based on training length.
    
    The "half-life" is how many steps until old weights contribute ~50%.
    
    Args:
        total_steps: Total training steps
        target_half_life_steps: Desired half-life (default: total_steps/10)
        
    Returns:
        Recommended decay value
        
    Example:
        >>> decay = compute_optimal_decay(total_steps=10000)
        >>> print(f"Recommended decay: {decay}")  # ~0.9993
    """
    import math
    
    if target_half_life_steps is None:
        # Default: half-life is 10% of total training
        target_half_life_steps = max(100, total_steps // 10)
    
    # decay^half_life = 0.5
    # half_life * log(decay) = log(0.5)
    # log(decay) = log(0.5) / half_life
    # decay = exp(log(0.5) / half_life)
    
    decay = math.exp(math.log(0.5) / target_half_life_steps)
    
    # Clamp to reasonable range
    decay = max(0.9, min(0.9999, decay))
    
    return decay


def print_ema_info(
    decay: float,
    total_steps: int
):
    """
    Print information about EMA configuration.
    
    Args:
        decay: EMA decay value
        total_steps: Total training steps
    """
    import math
    
    # Calculate half-life
    half_life = math.log(0.5) / math.log(decay)
    
    # Calculate contribution of initial weights at end
    initial_contribution = decay ** total_steps * 100
    
    print("\n" + "=" * 50)
    print("EMA CONFIGURATION")
    print("=" * 50)
    print(f"Decay:                    {decay}")
    print(f"Half-life:                {half_life:.0f} steps")
    print(f"Total steps:              {total_steps}")
    print(f"Initial weights at end:   {initial_contribution:.4f}%")
    print(f"Update per step:          {(1-decay)*100:.4f}%")
    print("=" * 50 + "\n")
