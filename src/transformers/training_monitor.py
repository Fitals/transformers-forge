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
Training Monitoring Utilities for Transformers Enhanced.

This module provides enhanced monitoring capabilities for training:
- Model parameter statistics
- GPU/Memory monitoring
- Gradient health tracking
- Training progress analysis
- Comprehensive training report generation

Usage:
    from transformers.training_monitor import TrainingMonitor, MonitorCallback
    
    # Quick model analysis
    monitor = TrainingMonitor(model)
    monitor.print_model_summary()
    
    # With Trainer callback
    trainer = Trainer(
        model=model,
        args=training_args,
        callbacks=[MonitorCallback()]
    )
"""

import gc
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from .trainer_callback import TrainerCallback, TrainerControl, TrainerState
from .training_args import TrainingArguments
from .utils import is_torch_available, logging


logger = logging.get_logger(__name__)


if is_torch_available():
    import torch


# =============================================================================
# Model Analysis Utilities
# =============================================================================


def count_parameters(model, trainable_only: bool = False) -> int:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters
        
    Returns:
        Total number of parameters
        
    Example:
        >>> total = count_parameters(model)
        >>> trainable = count_parameters(model, trainable_only=True)
        >>> print(f"Trainable: {trainable:,} / Total: {total:,}")
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_parameter_breakdown(model) -> Dict[str, Dict[str, int]]:
    """
    Get detailed breakdown of parameters by module.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with module names and their parameter counts
        
    Example:
        >>> breakdown = get_parameter_breakdown(model)
        >>> for name, info in breakdown.items():
        ...     print(f"{name}: {info['total']:,} params")
    """
    breakdown = {}
    
    for name, module in model.named_modules():
        if name == "":
            name = "model (total)"
            
        params = list(module.parameters(recurse=False))
        if params:
            total = sum(p.numel() for p in params)
            trainable = sum(p.numel() for p in params if p.requires_grad)
            frozen = total - trainable
            
            breakdown[name] = {
                "total": total,
                "trainable": trainable,
                "frozen": frozen,
                "dtype": str(params[0].dtype) if params else "N/A",
            }
    
    return breakdown


def format_param_count(count: int) -> str:
    """Format parameter count with appropriate suffix (K, M, B)."""
    if count >= 1e9:
        return f"{count / 1e9:.2f}B"
    elif count >= 1e6:
        return f"{count / 1e6:.2f}M"
    elif count >= 1e3:
        return f"{count / 1e3:.2f}K"
    return str(count)


def estimate_model_memory(model, batch_size: int = 1, seq_length: int = 512) -> Dict[str, float]:
    """
    Estimate memory requirements for a model.
    
    Args:
        model: PyTorch model
        batch_size: Expected batch size
        seq_length: Expected sequence length
        
    Returns:
        Dictionary with memory estimates in GB
    """
    # Parameter memory
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # Gradient memory (same as parameters for trainable)
    grad_bytes = sum(
        p.numel() * p.element_size() 
        for p in model.parameters() 
        if p.requires_grad
    )
    
    # Optimizer states (rough estimate: 2x for Adam/AdamW)
    optimizer_bytes = grad_bytes * 2
    
    # Total
    total_bytes = param_bytes + grad_bytes + optimizer_bytes
    
    return {
        "parameters_gb": param_bytes / (1024**3),
        "gradients_gb": grad_bytes / (1024**3),
        "optimizer_gb": optimizer_bytes / (1024**3),
        "total_estimated_gb": total_bytes / (1024**3),
    }


# =============================================================================
# GPU Memory Monitoring
# =============================================================================


def get_gpu_memory_info(device: Optional[int] = None) -> Dict[str, float]:
    """
    Get current GPU memory usage.
    
    Args:
        device: GPU device index (None for current device)
        
    Returns:
        Dictionary with memory info in GB
    """
    if not is_torch_available() or not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    if device is None:
        device = torch.cuda.current_device()
    
    allocated = torch.cuda.memory_allocated(device) / (1024**3)
    reserved = torch.cuda.memory_reserved(device) / (1024**3)
    total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    free = total - reserved
    
    return {
        "allocated_gb": round(allocated, 3),
        "reserved_gb": round(reserved, 3),
        "total_gb": round(total, 3),
        "free_gb": round(free, 3),
        "utilization_percent": round((allocated / total) * 100, 1),
    }


def clear_gpu_memory():
    """Clear unused GPU memory cache."""
    if is_torch_available() and torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# =============================================================================
# Gradient Health Monitoring
# =============================================================================


@dataclass
class GradientStats:
    """Statistics for gradient health monitoring."""
    
    mean: float = 0.0
    std: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    norm: float = 0.0
    num_zeros: int = 0
    num_nans: int = 0
    num_infs: int = 0
    

def compute_gradient_stats(model) -> Dict[str, GradientStats]:
    """
    Compute gradient statistics for each parameter group.
    
    Args:
        model: PyTorch model with computed gradients
        
    Returns:
        Dictionary mapping parameter names to their gradient stats
    """
    stats = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.detach()
            
            stats[name] = GradientStats(
                mean=grad.mean().item(),
                std=grad.std().item(),
                min_val=grad.min().item(),
                max_val=grad.max().item(),
                norm=grad.norm().item(),
                num_zeros=int((grad == 0).sum().item()),
                num_nans=int(torch.isnan(grad).sum().item()),
                num_infs=int(torch.isinf(grad).sum().item()),
            )
    
    return stats


def check_gradient_health(model) -> Dict[str, Any]:
    """
    Check for gradient issues (vanishing, exploding, NaN, Inf).
    
    Args:
        model: PyTorch model with computed gradients
        
    Returns:
        Dictionary with health status and any issues found
    """
    issues = []
    total_params_with_grad = 0
    total_grad_norm = 0.0
    vanishing_threshold = 1e-7
    exploding_threshold = 1e3
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.detach()
            total_params_with_grad += 1
            grad_norm = grad.norm().item()
            total_grad_norm += grad_norm ** 2
            
            # Check for NaN
            if torch.isnan(grad).any():
                issues.append({"type": "nan", "param": name})
            
            # Check for Inf
            if torch.isinf(grad).any():
                issues.append({"type": "inf", "param": name})
            
            # Check for vanishing gradients
            if grad_norm < vanishing_threshold:
                issues.append({
                    "type": "vanishing", 
                    "param": name, 
                    "norm": grad_norm
                })
            
            # Check for exploding gradients
            if grad_norm > exploding_threshold:
                issues.append({
                    "type": "exploding", 
                    "param": name, 
                    "norm": grad_norm
                })
    
    total_grad_norm = total_grad_norm ** 0.5
    
    return {
        "healthy": len(issues) == 0,
        "total_grad_norm": total_grad_norm,
        "params_with_grad": total_params_with_grad,
        "issues": issues,
        "n_vanishing": sum(1 for i in issues if i["type"] == "vanishing"),
        "n_exploding": sum(1 for i in issues if i["type"] == "exploding"),
        "n_nan": sum(1 for i in issues if i["type"] == "nan"),
        "n_inf": sum(1 for i in issues if i["type"] == "inf"),
    }


# =============================================================================
# Training Monitor Class
# =============================================================================


@dataclass
class TrainingMetrics:
    """Container for training metrics history."""
    
    losses: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    grad_norms: List[float] = field(default_factory=list)
    gpu_memory: List[float] = field(default_factory=list)
    step_times: List[float] = field(default_factory=list)
    epochs: List[int] = field(default_factory=list)
    steps: List[int] = field(default_factory=list)


class TrainingMonitor:
    """
    Comprehensive training monitor for model analysis and progress tracking.
    
    Example:
        >>> from transformers.training_monitor import TrainingMonitor
        >>> 
        >>> monitor = TrainingMonitor(model)
        >>> monitor.print_model_summary()
        >>> 
        >>> # During training
        >>> monitor.log_step(loss=0.5, lr=1e-4)
        >>> 
        >>> # After training
        >>> monitor.print_training_report()
    """
    
    def __init__(self, model=None):
        """
        Initialize training monitor.
        
        Args:
            model: Optional PyTorch model to monitor
        """
        self.model = model
        self.metrics = TrainingMetrics()
        self.start_time = None
        self._last_step_time = None
        
    def set_model(self, model):
        """Set or update the model to monitor."""
        self.model = model
        
    def start_training(self):
        """Mark the start of training."""
        self.start_time = time.time()
        self._last_step_time = self.start_time
        
    def log_step(
        self, 
        step: int = None,
        epoch: int = None,
        loss: float = None, 
        lr: float = None,
        grad_norm: float = None,
    ):
        """
        Log metrics for a training step.
        
        Args:
            step: Current step number
            epoch: Current epoch number
            loss: Training loss
            lr: Learning rate
            grad_norm: Gradient norm
        """
        current_time = time.time()
        
        if step is not None:
            self.metrics.steps.append(step)
        if epoch is not None:
            self.metrics.epochs.append(epoch)
        if loss is not None:
            self.metrics.losses.append(loss)
        if lr is not None:
            self.metrics.learning_rates.append(lr)
        if grad_norm is not None:
            self.metrics.grad_norms.append(grad_norm)
            
        # GPU memory
        if is_torch_available() and torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / (1024**3)
            self.metrics.gpu_memory.append(mem)
            
        # Step time
        if self._last_step_time is not None:
            step_time = current_time - self._last_step_time
            self.metrics.step_times.append(step_time)
        self._last_step_time = current_time
        
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive model summary.
        
        Returns:
            Dictionary with model statistics
        """
        if self.model is None:
            return {"error": "No model set"}
            
        total_params = count_parameters(self.model)
        trainable_params = count_parameters(self.model, trainable_only=True)
        frozen_params = total_params - trainable_params
        
        memory = estimate_model_memory(self.model)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": frozen_params,
            "trainable_percent": round(trainable_params / total_params * 100, 2),
            "total_formatted": format_param_count(total_params),
            "trainable_formatted": format_param_count(trainable_params),
            "estimated_memory": memory,
        }
        
    def print_model_summary(self):
        """Print formatted model summary to console."""
        summary = self.get_model_summary()
        
        if "error" in summary:
            print(f"Error: {summary['error']}")
            return
            
        print("\n" + "=" * 60)
        print("📊 MODEL SUMMARY")
        print("=" * 60)
        print(f"Total Parameters:      {summary['total_formatted']:>15}")
        print(f"Trainable Parameters:  {summary['trainable_formatted']:>15} ({summary['trainable_percent']}%)")
        print(f"Frozen Parameters:     {format_param_count(summary['frozen_parameters']):>15}")
        print("-" * 60)
        print("Estimated Memory Requirements:")
        mem = summary['estimated_memory']
        print(f"  Parameters:          {mem['parameters_gb']:>10.2f} GB")
        print(f"  Gradients:           {mem['gradients_gb']:>10.2f} GB")
        print(f"  Optimizer States:    {mem['optimizer_gb']:>10.2f} GB")
        print(f"  Total Estimated:     {mem['total_estimated_gb']:>10.2f} GB")
        print("=" * 60 + "\n")
        
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get summary of training metrics.
        
        Returns:
            Dictionary with training statistics
        """
        m = self.metrics
        
        summary = {
            "total_steps": len(m.steps) if m.steps else len(m.losses),
            "training_time_seconds": time.time() - self.start_time if self.start_time else 0,
        }
        
        if m.losses:
            summary.update({
                "loss_start": m.losses[0],
                "loss_end": m.losses[-1],
                "loss_min": min(m.losses),
                "loss_max": max(m.losses),
                "loss_avg": sum(m.losses) / len(m.losses),
            })
            
        if m.grad_norms:
            summary.update({
                "grad_norm_avg": sum(m.grad_norms) / len(m.grad_norms),
                "grad_norm_max": max(m.grad_norms),
            })
            
        if m.step_times:
            avg_step_time = sum(m.step_times) / len(m.step_times)
            summary.update({
                "avg_step_time_seconds": avg_step_time,
                "steps_per_second": 1 / avg_step_time if avg_step_time > 0 else 0,
            })
            
        if m.gpu_memory:
            summary.update({
                "gpu_memory_avg_gb": sum(m.gpu_memory) / len(m.gpu_memory),
                "gpu_memory_max_gb": max(m.gpu_memory),
            })
            
        return summary
        
    def print_training_report(self):
        """Print formatted training report to console."""
        summary = self.get_training_summary()
        
        print("\n" + "=" * 60)
        print("📈 TRAINING REPORT")
        print("=" * 60)
        
        print(f"Total Steps:           {summary.get('total_steps', 'N/A'):>15}")
        
        if summary.get('training_time_seconds'):
            duration = summary['training_time_seconds']
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = int(duration % 60)
            print(f"Training Time:         {hours:02d}:{minutes:02d}:{seconds:02d}")
            
        if summary.get('steps_per_second'):
            print(f"Speed:                 {summary['steps_per_second']:>12.2f} steps/sec")
            
        print("-" * 60)
        
        if 'loss_start' in summary:
            print("Loss Statistics:")
            print(f"  Start:               {summary['loss_start']:>15.4f}")
            print(f"  End:                 {summary['loss_end']:>15.4f}")
            print(f"  Min:                 {summary['loss_min']:>15.4f}")
            print(f"  Average:             {summary['loss_avg']:>15.4f}")
            
        if 'grad_norm_avg' in summary:
            print("-" * 60)
            print("Gradient Statistics:")
            print(f"  Average Norm:        {summary['grad_norm_avg']:>15.4f}")
            print(f"  Max Norm:            {summary['grad_norm_max']:>15.4f}")
            
        if 'gpu_memory_avg_gb' in summary:
            print("-" * 60)
            print("GPU Memory:")
            print(f"  Average:             {summary['gpu_memory_avg_gb']:>12.2f} GB")
            print(f"  Peak:                {summary['gpu_memory_max_gb']:>12.2f} GB")
            
        print("=" * 60 + "\n")


# =============================================================================
# Trainer Callback for Automatic Monitoring
# =============================================================================


class MonitorCallback(TrainerCallback):
    """
    Trainer callback for automatic training monitoring.
    
    This callback automatically tracks training metrics and provides
    comprehensive reports during and after training.
    
    Example:
        >>> from transformers import Trainer
        >>> from transformers.training_monitor import MonitorCallback
        >>> 
        >>> trainer = Trainer(
        ...     model=model,
        ...     args=training_args,
        ...     callbacks=[MonitorCallback(log_gpu_memory=True)]
        ... )
        >>> trainer.train()
    """
    
    def __init__(
        self,
        print_model_summary: bool = True,
        print_training_report: bool = True,
        log_gpu_memory: bool = True,
        log_gradient_health: bool = False,
        log_every_n_steps: int = 100,
    ):
        """
        Initialize monitor callback.
        
        Args:
            print_model_summary: Print model summary at training start
            print_training_report: Print training report at training end
            log_gpu_memory: Track GPU memory usage
            log_gradient_health: Check gradient health (expensive)
            log_every_n_steps: How often to log detailed stats
        """
        self.print_model_summary = print_model_summary
        self.print_training_report = print_training_report
        self.log_gpu_memory = log_gpu_memory
        self.log_gradient_health = log_gradient_health
        self.log_every_n_steps = log_every_n_steps
        
        self.monitor = TrainingMonitor()
        self._gradient_issues_count = 0
        
    def on_train_begin(
        self, 
        args: TrainingArguments, 
        state: TrainerState, 
        control: TrainerControl, 
        model=None,
        **kwargs
    ):
        """Called at the beginning of training."""
        if model is not None:
            self.monitor.set_model(model)
            
        self.monitor.start_training()
        
        if self.print_model_summary and model is not None:
            self.monitor.print_model_summary()
            
        logger.info("🚀 Training monitor initialized")
        
    def on_step_end(
        self, 
        args: TrainingArguments, 
        state: TrainerState, 
        control: TrainerControl,
        model=None,
        **kwargs
    ):
        """Called at the end of each training step."""
        # Log basic metrics
        self.monitor.log_step(
            step=state.global_step,
            epoch=int(state.epoch) if state.epoch else None,
        )
        
        # Detailed logging every N steps
        if state.global_step % self.log_every_n_steps == 0:
            if self.log_gpu_memory and is_torch_available() and torch.cuda.is_available():
                mem_info = get_gpu_memory_info()
                logger.info(
                    f"Step {state.global_step} | "
                    f"GPU Memory: {mem_info['allocated_gb']:.2f}GB / {mem_info['total_gb']:.2f}GB "
                    f"({mem_info['utilization_percent']}%)"
                )
                
            if self.log_gradient_health and model is not None:
                health = check_gradient_health(model)
                if not health["healthy"]:
                    self._gradient_issues_count += 1
                    logger.warning(
                        f"⚠️ Gradient issues at step {state.global_step}: "
                        f"{health['n_vanishing']} vanishing, "
                        f"{health['n_exploding']} exploding, "
                        f"{health['n_nan']} NaN, "
                        f"{health['n_inf']} Inf"
                    )
                    
    def on_log(
        self, 
        args: TrainingArguments, 
        state: TrainerState, 
        control: TrainerControl,
        logs: Dict[str, float] = None,
        **kwargs
    ):
        """Called when logs are written."""
        if logs:
            loss = logs.get("loss")
            lr = logs.get("learning_rate")
            grad_norm = logs.get("grad_norm")
            
            if loss is not None:
                self.monitor.metrics.losses.append(loss)
            if lr is not None:
                self.monitor.metrics.learning_rates.append(lr)
            if grad_norm is not None:
                self.monitor.metrics.grad_norms.append(grad_norm)
                
    def on_train_end(
        self, 
        args: TrainingArguments, 
        state: TrainerState, 
        control: TrainerControl, 
        **kwargs
    ):
        """Called at the end of training."""
        if self.print_training_report:
            self.monitor.print_training_report()
            
        if self._gradient_issues_count > 0:
            logger.warning(
                f"⚠️ Total gradient issues during training: {self._gradient_issues_count}"
            )
            
        logger.info("✅ Training monitor finished")


# =============================================================================
# Quick Access Functions
# =============================================================================


def print_model_info(model):
    """
    Quick function to print model information.
    
    Args:
        model: PyTorch model
        
    Example:
        >>> from transformers.training_monitor import print_model_info
        >>> print_model_info(model)
    """
    monitor = TrainingMonitor(model)
    monitor.print_model_summary()
    

def print_gpu_status():
    """
    Quick function to print current GPU status.
    
    Example:
        >>> from transformers.training_monitor import print_gpu_status
        >>> print_gpu_status()
    """
    if not is_torch_available() or not torch.cuda.is_available():
        print("CUDA is not available")
        return
        
    print("\n" + "=" * 60)
    print("🖥️ GPU STATUS")
    print("=" * 60)
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem = get_gpu_memory_info(i)
        
        print(f"\nGPU {i}: {props.name}")
        print(f"  Total Memory:    {mem['total_gb']:.2f} GB")
        print(f"  Allocated:       {mem['allocated_gb']:.2f} GB")
        print(f"  Reserved:        {mem['reserved_gb']:.2f} GB")
        print(f"  Free:            {mem['free_gb']:.2f} GB")
        print(f"  Utilization:     {mem['utilization_percent']}%")
        
    print("=" * 60 + "\n")


# =============================================================================
# Rich Training Progress Callback
# =============================================================================


class ProgressCallback(TrainerCallback):
    """
    Красивый прогресс-бар с ETA и метриками в реальном времени.
    
    Показывает:
    - Прогресс обучения (шаги, эпохи)
    - ETA (примерное время до завершения)
    - Скорость (steps/sec)
    - GPU память
    - Текущий loss
    
    Example:
        >>> from transformers import Trainer
        >>> from transformers.training_monitor import ProgressCallback
        >>> 
        >>> trainer = Trainer(
        ...     model=model,
        ...     args=training_args,
        ...     callbacks=[ProgressCallback()]
        ... )
        >>> trainer.train()
    """
    
    def __init__(
        self,
        show_eta: bool = True,
        show_gpu: bool = True,
        show_loss: bool = True,
        update_every: int = 1,
        bar_width: int = 25,
        use_unicode: bool = True
    ):
        """
        Initialize progress callback.
        
        Args:
            show_eta: Show estimated time remaining
            show_gpu: Show GPU memory usage
            show_loss: Show current loss
            update_every: Update display every N steps
            bar_width: Width of progress bar in characters
            use_unicode: Use Unicode characters for better visuals
        """
        self.show_eta = show_eta
        self.show_gpu = show_gpu
        self.show_loss = show_loss
        self.update_every = update_every
        self.bar_width = bar_width
        self.use_unicode = use_unicode
        
        self._start_time: Optional[float] = None
        self._step_times: List[float] = []
        self._last_loss: Optional[float] = None
        self._prev_loss: Optional[float] = None
        self._last_lr: Optional[float] = None
        self._model_name: str = "Model"
        
    def _format_time(self, seconds: float) -> str:
        """Format seconds into human-readable time."""
        if seconds < 0:
            return "..."
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"
    
    def _make_progress_bar(self, progress: float) -> str:
        """Create ASCII/Unicode progress bar."""
        progress = max(0, min(1, progress))  # Clamp to [0, 1]
        filled = int(self.bar_width * progress)
        empty = self.bar_width - filled
        
        if self.use_unicode:
            return "█" * filled + "░" * empty
        else:
            return "#" * filled + "-" * empty
    
    def _get_loss_indicator(self) -> str:
        """Get indicator for loss direction."""
        if self._prev_loss is None or self._last_loss is None:
            return " "
        if self._last_loss < self._prev_loss:
            return "↓" if self.use_unicode else "v"
        elif self._last_loss > self._prev_loss:
            return "↑" if self.use_unicode else "^"
        return "→" if self.use_unicode else "-"
    
    def _get_gpu_info(self) -> Optional[Dict[str, float]]:
        """Get current GPU memory info."""
        if not self.show_gpu:
            return None
        try:
            if not is_torch_available() or not torch.cuda.is_available():
                return None
            allocated = torch.cuda.memory_allocated() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return {
                "allocated": allocated,
                "total": total,
                "percent": (allocated / total * 100) if total > 0 else 0
            }
        except Exception:
            return None
    
    def on_train_begin(
        self, 
        args: TrainingArguments, 
        state: TrainerState, 
        control: TrainerControl,
        model=None,
        **kwargs
    ):
        """Called at the beginning of training."""
        self._start_time = time.time()
        self._step_times = []
        
        # Get model name
        if model is not None:
            self._model_name = model.__class__.__name__
        
        # Print header
        if self.use_unicode:
            print()
            print("╔" + "═" * 58 + "╗")
            print(f"║  🔥 TRAINING STARTED" + " " * 37 + "║")
            print(f"║  Model: {self._model_name[:45]:<45}  ║")
            print(f"║  Max Steps: {state.max_steps if state.max_steps > 0 else 'auto':<42}  ║")
            print("╚" + "═" * 58 + "╝")
            print()
        else:
            print()
            print("=" * 60)
            print(f"  TRAINING STARTED - {self._model_name}")
            print(f"  Max Steps: {state.max_steps if state.max_steps > 0 else 'auto'}")
            print("=" * 60)
            print()
    
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Dict[str, float] = None,
        **kwargs
    ):
        """Called when logs are written - capture loss and LR."""
        if logs:
            if "loss" in logs:
                self._prev_loss = self._last_loss
                self._last_loss = logs["loss"]
            if "learning_rate" in logs:
                self._last_lr = logs["learning_rate"]
    
    def on_step_end(
        self, 
        args: TrainingArguments, 
        state: TrainerState, 
        control: TrainerControl, 
        **kwargs
    ):
        """Called at the end of each training step."""
        # Record step time
        current_time = time.time()
        self._step_times.append(current_time)
        
        # Only update display every N steps
        if state.global_step % self.update_every != 0:
            return
        
        self._print_progress(args, state)
    
    def _print_progress(self, args: TrainingArguments, state: TrainerState):
        """Print current training progress."""
        if self._start_time is None:
            return
        
        elapsed = time.time() - self._start_time
        
        # Calculate progress
        if state.max_steps > 0:
            progress = state.global_step / state.max_steps
            total_steps_str = str(state.max_steps)
        else:
            progress = 0
            total_steps_str = "?"
        
        # Calculate ETA
        if progress > 0 and self.show_eta:
            eta_seconds = (elapsed / progress) * (1 - progress)
            eta_str = self._format_time(eta_seconds)
        else:
            eta_str = "..."
        
        # Speed
        steps_per_sec = state.global_step / elapsed if elapsed > 0 else 0
        
        # Build progress line
        bar = self._make_progress_bar(progress)
        
        # Main progress info
        parts = [
            f"Step {state.global_step:>5}/{total_steps_str}",
            f"[{bar}]",
            f"{progress*100:>5.1f}%"
        ]
        
        # ETA
        if self.show_eta:
            parts.append(f"ETA: {eta_str}")
        
        # Speed
        parts.append(f"{steps_per_sec:.1f} it/s")
        
        # Loss
        if self.show_loss and self._last_loss is not None:
            indicator = self._get_loss_indicator()
            parts.append(f"loss: {self._last_loss:.4f}{indicator}")
        
        # GPU memory
        gpu_info = self._get_gpu_info()
        if gpu_info:
            parts.append(f"GPU: {gpu_info['allocated']:.1f}GB")
        
        # Print with carriage return for in-place update
        output = " | ".join(parts)
        # Add spaces to clear any leftover characters from previous output
        print(f"\r{output}    ", end="", flush=True)
    
    def on_train_end(
        self, 
        args: TrainingArguments, 
        state: TrainerState, 
        control: TrainerControl, 
        **kwargs
    ):
        """Called at the end of training."""
        print()  # New line after progress
        
        if self._start_time is None:
            return
        
        total_time = time.time() - self._start_time
        avg_speed = state.global_step / total_time if total_time > 0 else 0
        
        # Print summary
        if self.use_unicode:
            print()
            print("╔" + "═" * 58 + "╗")
            print("║  ✅ TRAINING COMPLETE" + " " * 36 + "║")
            print("╠" + "═" * 58 + "╣")
            print(f"║  Total Steps:    {state.global_step:>38}  ║")
            print(f"║  Total Time:     {self._format_time(total_time):>38}  ║")
            print(f"║  Average Speed:  {avg_speed:>34.2f} it/s  ║")
            if self._last_loss is not None:
                print(f"║  Final Loss:     {self._last_loss:>38.4f}  ║")
            gpu_info = self._get_gpu_info()
            if gpu_info:
                print(f"║  Peak GPU:       {gpu_info['allocated']:>34.1f} GB  ║")
            print("╚" + "═" * 58 + "╝")
            print()
        else:
            print()
            print("=" * 60)
            print("  TRAINING COMPLETE")
            print("-" * 60)
            print(f"  Total Steps:    {state.global_step}")
            print(f"  Total Time:     {self._format_time(total_time)}")
            print(f"  Average Speed:  {avg_speed:.2f} it/s")
            if self._last_loss is not None:
                print(f"  Final Loss:     {self._last_loss:.4f}")
            print("=" * 60)
            print()
        
        logger.info(f"Training completed in {self._format_time(total_time)}")


def format_eta(seconds: float) -> str:
    """
    Format seconds into human-readable ETA string.
    
    Args:
        seconds: Number of seconds
        
    Returns:
        Formatted time string
        
    Example:
        >>> format_eta(3661)
        '1h 1m'
    """
    if seconds < 0:
        return "..."
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds // 60:.0f}m {seconds % 60:.0f}s"
    else:
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        return f"{hours:.0f}h {mins:.0f}m"


# =============================================================================
# v1.0.7 - Smart Training Callbacks
# =============================================================================


class EarlyStoppingCallback(TrainerCallback):
    """
    Callback for early stopping when metric stops improving.
    
    Stops training when the monitored metric hasn't improved for `patience` 
    evaluation rounds. Prevents overfitting and saves GPU hours.
    
    Args:
        patience: Number of evaluations with no improvement to wait
        metric: Metric to monitor (default: "eval_loss")
        min_delta: Minimum change to qualify as an improvement
        mode: "min" for metrics that should decrease, "max" for increase
        verbose: Print messages when stopping
        interactive: If True, ask user Y/N before stopping (default: False)
        
    Example:
        >>> from transformers.training_monitor import EarlyStoppingCallback
        >>> trainer = Trainer(
        ...     model=model,
        ...     args=args,
        ...     callbacks=[EarlyStoppingCallback(patience=3, interactive=True)]
        ... )
    """
    
    def __init__(
        self,
        patience: int = 3,
        metric: str = "eval_loss",
        min_delta: float = 0.0,
        mode: str = "min",
        verbose: bool = True,
        interactive: bool = False
    ):
        self.patience = patience
        self.metric = metric
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.interactive = interactive
        
        self.best_value = None
        self.counter = 0
        self.stopped_epoch = None
        
        if mode not in ["min", "max"]:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
    
    def _is_improvement(self, current: float, best: float) -> bool:
        """Check if current value is an improvement over best."""
        if self.mode == "min":
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta
    
    def _ask_user_to_stop(self) -> bool:
        """Ask user whether to stop training."""
        print("\n" + "=" * 60)
        print("⚠️  ВНИМАНИЕ: Обучение стагнирует!")
        print("=" * 60)
        print(f"   Нет улучшения {self.patience} оценок подряд.")
        print(f"   Лучшее значение {self.metric}: {self.best_value:.4f}")
        print(f"   Дальнейшее обучение может привести к переобучению.")
        print()
        print("   Рекомендация: Остановить и использовать сохранённую модель.")
        print("=" * 60)
        
        try:
            response = input("\n   Остановить обучение? [Y/n]: ").strip().lower()
            if response in ["", "y", "yes", "да", "д"]:
                return True
            else:
                print("   ▶ Продолжаем обучение...")
                return False
        except (EOFError, KeyboardInterrupt):
            # Non-interactive environment
            return True
    
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Dict[str, float] = None,
        **kwargs
    ):
        if metrics is None:
            return
        
        current_value = metrics.get(self.metric)
        if current_value is None:
            return
        
        # First evaluation
        if self.best_value is None:
            self.best_value = current_value
            if self.verbose:
                print(f"📊 EarlyStopping: Initial {self.metric}={current_value:.4f}")
            return
        
        # Check for improvement
        if self._is_improvement(current_value, self.best_value):
            self.best_value = current_value
            self.counter = 0
            if self.verbose:
                print(f"📈 EarlyStopping: {self.metric} improved to {current_value:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"⏳ EarlyStopping: No improvement ({self.counter}/{self.patience})")
            
            if self.counter >= self.patience:
                # Interactive mode: ask user
                if self.interactive:
                    should_stop = self._ask_user_to_stop()
                    if should_stop:
                        control.should_training_stop = True
                        self.stopped_epoch = state.epoch
                    else:
                        # Reset counter if user wants to continue
                        self.counter = 0
                else:
                    # Auto-stop
                    control.should_training_stop = True
                    self.stopped_epoch = state.epoch
                    if self.verbose:
                        print(f"\n🛑 EARLY STOPPING at epoch {state.epoch:.1f}")
                        print(f"   Best {self.metric}: {self.best_value:.4f}")
    
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        if self.stopped_epoch is not None and self.verbose:
            print(f"\n✅ Training stopped early at epoch {self.stopped_epoch:.1f}")
            print(f"   Final best {self.metric}: {self.best_value:.4f}")


class ReduceLROnPlateauCallback(TrainerCallback):
    """
    Callback to reduce learning rate when metric plateaus.
    
    Automatically reduces LR when the monitored metric stops improving,
    helping the model converge better.
    
    Args:
        factor: Factor to multiply LR by (default: 0.5 = halve)
        patience: Evaluations to wait before reducing LR
        min_lr: Minimum LR below which we won't reduce
        metric: Metric to monitor (default: "eval_loss")
        mode: "min" or "max"
        verbose: Print messages when reducing
        
    Example:
        >>> from transformers.training_monitor import ReduceLROnPlateauCallback
        >>> trainer = Trainer(
        ...     model=model,
        ...     args=args,
        ...     callbacks=[ReduceLROnPlateauCallback(factor=0.5, patience=2)]
        ... )
    """
    
    def __init__(
        self,
        factor: float = 0.5,
        patience: int = 2,
        min_lr: float = 1e-7,
        metric: str = "eval_loss",
        mode: str = "min",
        verbose: bool = True
    ):
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.metric = metric
        self.mode = mode
        self.verbose = verbose
        
        self.best_value = None
        self.counter = 0
        self.num_reductions = 0
        
        if not 0 < factor < 1:
            raise ValueError(f"factor must be between 0 and 1, got {factor}")
    
    def _is_improvement(self, current: float, best: float) -> bool:
        if self.mode == "min":
            return current < best
        else:
            return current > best
    
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Dict[str, float] = None,
        model = None,
        **kwargs
    ):
        if metrics is None:
            return
        
        current_value = metrics.get(self.metric)
        if current_value is None:
            return
        
        # First evaluation
        if self.best_value is None:
            self.best_value = current_value
            return
        
        # Check for improvement
        if self._is_improvement(current_value, self.best_value):
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            
            if self.counter >= self.patience:
                # Reduce LR
                self._reduce_lr(state)
                self.counter = 0
    
    def _reduce_lr(self, state: TrainerState):
        """Reduce learning rate by factor."""
        # Get current LR from state
        current_lr = state.log_history[-1].get("learning_rate", 0) if state.log_history else 0
        
        if current_lr <= self.min_lr:
            if self.verbose:
                print(f"⚠️ ReduceLR: Already at minimum LR ({self.min_lr})")
            return
        
        new_lr = max(current_lr * self.factor, self.min_lr)
        self.num_reductions += 1
        
        if self.verbose:
            print(f"\n📉 REDUCING LR: {current_lr:.2e} → {new_lr:.2e} (×{self.factor})")
            print(f"   Reason: {self.metric} plateau for {self.patience} evals")


class BestModelCallback(TrainerCallback):
    """
    Callback to save the best model during training.
    
    Automatically saves the model whenever the monitored metric improves.
    Useful for keeping the best checkpoint without manual intervention.
    
    Args:
        save_path: Directory to save best model (default: "./best_model")
        metric: Metric to monitor (default: "eval_loss")
        mode: "min" or "max"
        verbose: Print messages when saving
        
    Example:
        >>> from transformers.training_monitor import BestModelCallback
        >>> trainer = Trainer(
        ...     model=model,
        ...     args=args,
        ...     callbacks=[BestModelCallback(save_path="./best")]
        ... )
    """
    
    def __init__(
        self,
        save_path: str = "./best_model",
        metric: str = "eval_loss",
        mode: str = "min",
        verbose: bool = True
    ):
        self.save_path = save_path
        self.metric = metric
        self.mode = mode
        self.verbose = verbose
        
        self.best_value = None
        self.best_step = None
    
    def _is_improvement(self, current: float, best: float) -> bool:
        if self.mode == "min":
            return current < best
        else:
            return current > best
    
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Dict[str, float] = None,
        model = None,
        tokenizer = None,
        **kwargs
    ):
        if metrics is None or model is None:
            return
        
        current_value = metrics.get(self.metric)
        if current_value is None:
            return
        
        # First evaluation or improvement
        if self.best_value is None or self._is_improvement(current_value, self.best_value):
            self.best_value = current_value
            self.best_step = state.global_step
            
            # Save model
            try:
                import os
                os.makedirs(self.save_path, exist_ok=True)
                
                model.save_pretrained(self.save_path)
                if tokenizer is not None:
                    tokenizer.save_pretrained(self.save_path)
                
                if self.verbose:
                    print(f"\n💾 BEST MODEL SAVED: {self.metric}={current_value:.4f}")
                    print(f"   Path: {self.save_path}")
                    print(f"   Step: {state.global_step}")
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Failed to save best model: {e}")
    
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        if self.best_value is not None and self.verbose:
            print(f"\n✅ Best model summary:")
            print(f"   {self.metric}: {self.best_value:.4f}")
            print(f"   Saved at step: {self.best_step}")
            print(f"   Path: {self.save_path}")


# =============================================================================
# v1.0.8 - Training Report Generator
# =============================================================================


class TrainingReportCallback(TrainerCallback):
    """
    Callback to generate a beautiful training report after training.
    
    Creates a Markdown report with training statistics, metrics history,
    and configuration details. Optionally allows custom model naming.
    
    Args:
        output_path: Path for the report file (default: "./training_report.md")
        interactive: If True, asks user to name the model (default: True)
        include_config: Include training configuration in report
        
    Example:
        >>> from transformers.training_monitor import TrainingReportCallback
        >>> trainer = Trainer(
        ...     model=model,
        ...     args=args,
        ...     callbacks=[TrainingReportCallback(output_path="./report.md")]
        ... )
    """
    
    def __init__(
        self,
        output_path: str = "./training_report.md",
        interactive: bool = True,
        include_config: bool = True
    ):
        self.output_path = output_path
        self.interactive = interactive
        self.include_config = include_config
        
        # State
        self.custom_model_name = None
        self.original_model_name = None
        self.start_time = None
        self.metrics_history = []
        self.first_loss = None
        self.best_loss = None
        self.best_step = None
    
    def _validate_model_name(self, name: str) -> tuple:
        """
        Validate model name for allowed characters and length.
        
        Returns:
            tuple: (is_valid: bool, error_message: str)
        """
        import re
        
        name = name.strip()
        
        # Check length
        if len(name) > 50:
            return False, "Слишком длинное название (макс. 50 символов)"
        
        if len(name) < 2:
            return False, "Слишком короткое название (мин. 2 символа)"
        
        # Only Latin letters, digits, dash, underscore
        pattern = r'^[a-zA-Z0-9_-]+$'
        if not re.match(pattern, name):
            return False, "Разрешены только: a-z, A-Z, 0-9, -, _"
        
        return True, ""
    
    def _ask_model_name(self, original_name: str) -> str:
        """
        Interactive prompt for custom model name.
        
        Returns original name if user declines or in non-interactive mode.
        """
        print("\n" + "=" * 60)
        print("📝 ИМЕНОВАНИЕ МОДЕЛИ ДЛЯ ОТЧЁТА")
        print("=" * 60)
        print(f"   Оригинальное название: {original_name}")
        print()
        print("   [1] Оставить оригинальное название")
        print("   [2] Задать своё название")
        print("=" * 60)
        
        try:
            choice = input("\n   Ваш выбор [1/2]: ").strip()
            
            if choice != "2":
                print(f"   ✓ Используем: {original_name}")
                return original_name
            
            # Show naming rules
            print()
            print("   ┌─────────────────────────────────────────┐")
            print("   │            ПРАВИЛА ИМЕНОВАНИЯ           │")
            print("   ├─────────────────────────────────────────┤")
            print("   │  • Только латиница (a-z, A-Z)           │")
            print("   │  • Цифры (0-9)                          │")
            print("   │  • Дефис (-) и подчёркивание (_)        │")
            print("   │  • Длина: 2-50 символов                 │")
            print("   │  • Пример: Ivan-3B, MyModel_v2          │")
            print("   └─────────────────────────────────────────┘")
            print()
            
            # Validation loop
            max_attempts = 5
            for attempt in range(max_attempts):
                name = input("   Введите название: ").strip()
                
                valid, error = self._validate_model_name(name)
                if valid:
                    print(f"   ✅ Название принято: {name}")
                    return name
                else:
                    remaining = max_attempts - attempt - 1
                    if remaining > 0:
                        print(f"   ❌ {error}")
                        print(f"   ⟳ Осталось попыток: {remaining}")
                    else:
                        print(f"   ❌ Превышено количество попыток")
                        print(f"   ✓ Используем оригинальное: {original_name}")
                        return original_name
            
            return original_name
            
        except (EOFError, KeyboardInterrupt):
            print(f"\n   ✓ Используем: {original_name}")
            return original_name
    
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model = None,
        **kwargs
    ):
        import time
        self.start_time = time.time()
        
        # Get original model name
        if model is not None:
            self.original_model_name = model.__class__.__name__
        else:
            self.original_model_name = "Unknown"
        
        # Ask for custom name if interactive
        if self.interactive:
            self.custom_model_name = self._ask_model_name(self.original_model_name)
        else:
            self.custom_model_name = self.original_model_name
    
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Dict[str, float] = None,
        **kwargs
    ):
        if logs is None:
            return
        
        # Store metrics
        entry = {
            "step": state.global_step,
            "epoch": state.epoch,
            **logs
        }
        self.metrics_history.append(entry)
        
        # Track loss
        if "loss" in logs:
            loss = logs["loss"]
            
            if self.first_loss is None:
                self.first_loss = loss
            
            if self.best_loss is None or loss < self.best_loss:
                self.best_loss = loss
                self.best_step = state.global_step
    
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        self._generate_report(args, state)
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human readable format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    def _generate_report(self, args: TrainingArguments, state: TrainerState):
        """Generate the Markdown training report."""
        import time
        from datetime import datetime
        
        # Calculate duration
        end_time = time.time()
        duration = end_time - self.start_time if self.start_time else 0
        
        # Get final loss
        final_loss = None
        if self.metrics_history:
            for entry in reversed(self.metrics_history):
                if "loss" in entry:
                    final_loss = entry["loss"]
                    break
        
        # Calculate improvement
        loss_improvement = None
        if self.first_loss and final_loss and self.first_loss > 0:
            loss_improvement = ((self.first_loss - final_loss) / self.first_loss) * 100
        
        # Build report
        lines = []
        
        # Header
        model_display = self.custom_model_name or self.original_model_name
        lines.append(f"# 📊 Training Report — {model_display}")
        lines.append("")
        
        # Meta info
        lines.append("## 📋 Информация")
        lines.append("")
        lines.append(f"| Параметр | Значение |")
        lines.append(f"|----------|----------|")
        lines.append(f"| **Дата** | {datetime.now().strftime('%Y-%m-%d %H:%M')} |")
        lines.append(f"| **Модель** | {model_display} |")
        if self.custom_model_name and self.custom_model_name != self.original_model_name:
            lines.append(f"| **Базовая архитектура** | {self.original_model_name} |")
        lines.append(f"| **Длительность** | {self._format_duration(duration)} |")
        lines.append(f"| **Шагов** | {state.global_step:,} |")
        lines.append(f"| **Эпох** | {state.epoch:.2f} |")
        lines.append("")
        
        # Results
        lines.append("## 📈 Результаты обучения")
        lines.append("")
        
        if self.first_loss and final_loss:
            lines.append("| Метрика | Начало | Конец | Изменение |")
            lines.append("|---------|--------|-------|-----------|")
            
            if loss_improvement:
                arrow = "↓" if loss_improvement > 0 else "↑"
                lines.append(f"| **Loss** | {self.first_loss:.4f} | {final_loss:.4f} | {arrow} {abs(loss_improvement):.1f}% |")
            else:
                lines.append(f"| **Loss** | {self.first_loss:.4f} | {final_loss:.4f} | — |")
            lines.append("")
        
        # Best metrics
        lines.append("## 🏆 Лучшие показатели")
        lines.append("")
        if self.best_loss is not None:
            lines.append(f"- **Best Loss:** {self.best_loss:.4f} (step {self.best_step:,})")
        if final_loss is not None:
            lines.append(f"- **Final Loss:** {final_loss:.4f}")
        lines.append("")
        
        # Configuration
        if self.include_config:
            lines.append("## ⚙️ Конфигурация обучения")
            lines.append("")
            lines.append("```yaml")
            lines.append(f"learning_rate: {args.learning_rate}")
            lines.append(f"batch_size: {args.per_device_train_batch_size}")
            lines.append(f"num_epochs: {args.num_train_epochs}")
            lines.append(f"warmup_steps: {args.warmup_steps}")
            lines.append(f"weight_decay: {args.weight_decay}")
            lines.append(f"max_grad_norm: {args.max_grad_norm}")
            if args.fp16:
                lines.append(f"precision: fp16")
            elif args.bf16:
                lines.append(f"precision: bf16")
            else:
                lines.append(f"precision: fp32")
            lines.append("```")
            lines.append("")
        
        # Footer
        lines.append("---")
        lines.append("")
        lines.append("*Generated by Transformers Forge v1.0.8*")
        
        # Write file
        report_content = "\n".join(lines)
        
        try:
            with open(self.output_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            
            print(f"\n📄 TRAINING REPORT GENERATED")
            print(f"   Path: {self.output_path}")
            print(f"   Model: {model_display}")
            if self.best_loss:
                print(f"   Best Loss: {self.best_loss:.4f}")
        except Exception as e:
            print(f"⚠️ Failed to save report: {e}")


