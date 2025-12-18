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
