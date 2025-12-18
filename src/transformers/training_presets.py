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
Training Presets for Transformers Enhanced.

This module provides ready-to-use, optimized configurations for common 
LLM training scenarios. All presets are battle-tested and designed to
work out-of-the-box with minimal configuration.

Presets available:
- SFTPreset: Supervised Fine-Tuning
- LoRAPreset: LoRA fine-tuning (PEFT)
- QLoRAPreset: 4-bit Quantized LoRA
- DPOPreset: Direct Preference Optimization
- MemoryEfficientPreset: For limited GPU memory

Usage:
    from transformers.training_presets import get_preset, LoRAPreset
    
    # Quick start with defaults
    args = get_preset("lora").get_training_args()
    
    # Custom configuration
    preset = LoRAPreset(
        output_dir="./my_model",
        learning_rate=2e-4,
        lora_r=16
    )
    training_args = preset.get_training_args()
    lora_config = preset.get_lora_config()
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

from .training_args import TrainingArguments
from .utils import logging



logger = logging.get_logger(__name__)


# =============================================================================
# Configuration Validation
# =============================================================================


@dataclass
class ConfigIssue:
    """
    Представляет проблему в конфигурации обучения.
    
    Attributes:
        level: Уровень проблемы ("warning" или "error")
        message: Описание проблемы
        fix_suggestion: Рекомендация по исправлению
        auto_fixable: Можно ли автоматически исправить
        param_name: Имя параметра (для auto_fix)
        fixed_value: Значение для автоисправления
    """
    level: str  # "warning" или "error"
    message: str
    fix_suggestion: str
    auto_fixable: bool = True
    param_name: Optional[str] = None
    fixed_value: Any = None


# =============================================================================
# Base Preset Class
# =============================================================================


@dataclass
class BasePreset:
    """
    Base class for all training presets.
    
    Provides common parameters and utilities for creating
    TrainingArguments with sensible defaults.
    """
    
    # Core training parameters
    output_dir: str = field(default="./output")
    num_train_epochs: float = field(default=3.0)
    max_steps: int = field(default=-1)
    
    # Batch configuration
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=4)
    
    # Optimizer settings
    learning_rate: float = field(default=2e-5)
    weight_decay: float = field(default=0.01)
    warmup_ratio: float = field(default=0.1)
    warmup_steps: int = field(default=0)
    max_grad_norm: float = field(default=1.0)
    
    # Scheduler
    lr_scheduler_type: str = field(default="cosine")
    
    # Precision
    bf16: bool = field(default=True)
    fp16: bool = field(default=False)
    
    # Logging & Saving
    logging_steps: int = field(default=10)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=500)
    save_total_limit: int = field(default=3)
    
    # Evaluation
    eval_strategy: str = field(default="steps")
    eval_steps: int = field(default=500)
    
    # Memory optimization
    gradient_checkpointing: bool = field(default=False)
    
    # Seed for reproducibility
    seed: int = field(default=42)
    
    # Additional kwargs for TrainingArguments
    extra_args: Dict[str, Any] = field(default_factory=dict)
    
    def get_training_args(self) -> TrainingArguments:
        """
        Create TrainingArguments from this preset.
        
        Returns:
            TrainingArguments configured according to this preset
            
        Note:
            Automatically detects GPU availability and adjusts
            bf16/fp16 settings accordingly.
        """
        # Auto-detect precision support
        bf16_enabled = self.bf16
        fp16_enabled = self.fp16
        use_cpu = False
        
        try:
            import torch
            if not torch.cuda.is_available():
                # No GPU - disable bf16/fp16
                bf16_enabled = False
                fp16_enabled = False
                use_cpu = True
                logger.warning(
                    "No GPU detected. Disabling bf16/fp16 and using CPU. "
                    "Training will be slow."
                )
            elif bf16_enabled:
                # Check if bf16 is supported
                if not torch.cuda.is_bf16_supported():
                    logger.warning(
                        "bf16 not supported on this GPU. Falling back to fp16."
                    )
                    bf16_enabled = False
                    fp16_enabled = True
        except ImportError:
            # torch not available
            bf16_enabled = False
            fp16_enabled = False
            use_cpu = True
        
        args_dict = {
            "output_dir": self.output_dir,
            "num_train_epochs": self.num_train_epochs,
            "max_steps": self.max_steps,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "warmup_steps": self.warmup_steps,
            "max_grad_norm": self.max_grad_norm,
            "lr_scheduler_type": self.lr_scheduler_type,
            "bf16": bf16_enabled,
            "fp16": fp16_enabled,
            "logging_steps": self.logging_steps,
            "save_strategy": self.save_strategy,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "eval_strategy": self.eval_strategy,
            "eval_steps": self.eval_steps,
            "gradient_checkpointing": self.gradient_checkpointing,
            "seed": self.seed,
        }
        
        # Add use_cpu if needed
        if use_cpu:
            args_dict["use_cpu"] = True
        
        # Merge with extra_args
        args_dict.update(self.extra_args)
        
        return TrainingArguments(**args_dict)
    
    def get_effective_batch_size(self, num_gpus: int = 1) -> int:
        """
        Calculate effective batch size.
        
        Args:
            num_gpus: Number of GPUs used
            
        Returns:
            Total effective batch size per step
        """
        return (
            self.per_device_train_batch_size 
            * self.gradient_accumulation_steps 
            * num_gpus
        )
    
    def summary(self) -> str:
        """Get a summary of the preset configuration."""
        lines = [
            f"=== {self.__class__.__name__} ===",
            f"Output: {self.output_dir}",
            f"Epochs: {self.num_train_epochs}",
            f"Batch: {self.per_device_train_batch_size} x {self.gradient_accumulation_steps} = {self.get_effective_batch_size()}",
            f"LR: {self.learning_rate} ({self.lr_scheduler_type})",
            f"Precision: {'bf16' if self.bf16 else 'fp16' if self.fp16 else 'fp32'}",
            f"Gradient Checkpointing: {self.gradient_checkpointing}",
        ]
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return self.summary()
    
    def validate(self) -> List[ConfigIssue]:
        """
        Проверить конфигурацию на проблемы.
        
        Returns:
            Список найденных проблем
            
        Example:
            >>> preset = LoRAPreset(learning_rate=0.1)
            >>> issues = preset.validate()
            >>> for issue in issues:
            ...     print(f"{issue.level}: {issue.message}")
        """
        issues = []
        
        # Проверяем наличие CUDA
        try:
            import torch
            has_cuda = torch.cuda.is_available()
            has_bf16 = has_cuda and torch.cuda.is_bf16_supported()
        except ImportError:
            has_cuda = False
            has_bf16 = False
        
        # === Проверка precision ===
        
        if self.bf16 and not has_cuda:
            issues.append(ConfigIssue(
                level="warning",
                message="bf16=True но CUDA недоступен",
                fix_suggestion="Использовать fp32 на CPU",
                param_name="bf16",
                fixed_value=False
            ))
        
        if self.bf16 and has_cuda and not has_bf16:
            issues.append(ConfigIssue(
                level="warning",
                message="bf16 не поддерживается на этом GPU",
                fix_suggestion="Использовать fp16 вместо bf16",
                param_name="bf16",
                fixed_value=False
            ))
        
        if self.fp16 and not has_cuda:
            issues.append(ConfigIssue(
                level="warning",
                message="fp16=True но CUDA недоступен",
                fix_suggestion="Использовать fp32 на CPU",
                param_name="fp16",
                fixed_value=False
            ))
        
        # === Проверка learning rate ===
        
        if self.learning_rate > 1e-2:
            issues.append(ConfigIssue(
                level="warning",
                message=f"learning_rate={self.learning_rate} очень высокий",
                fix_suggestion="Для fine-tuning рекомендуется 1e-5 - 5e-4",
                param_name="learning_rate",
                fixed_value=2e-5
            ))
        
        if self.learning_rate < 1e-7:
            issues.append(ConfigIssue(
                level="warning",
                message=f"learning_rate={self.learning_rate} очень низкий",
                fix_suggestion="Обучение будет очень медленным",
                param_name="learning_rate",
                fixed_value=2e-5
            ))
        
        # === Проверка warmup ===
        
        if self.warmup_ratio > 0 and self.warmup_steps > 0:
            issues.append(ConfigIssue(
                level="warning",
                message="Заданы и warmup_ratio и warmup_steps",
                fix_suggestion="Используйте только один параметр",
                param_name="warmup_steps",
                fixed_value=0
            ))
        
        # === Проверка batch size ===
        
        effective_batch = self.get_effective_batch_size()
        if effective_batch > 128:
            issues.append(ConfigIssue(
                level="warning",
                message=f"Effective batch size={effective_batch} очень большой",
                fix_suggestion="Может привести к Out of Memory",
                auto_fixable=False
            ))
        
        return issues
    
    def auto_fix(self) -> List[str]:
        """
        Автоматически исправить проблемы в конфигурации.
        
        Returns:
            Список сделанных изменений
            
        Example:
            >>> preset = LoRAPreset(learning_rate=0.1)
            >>> changes = preset.auto_fix()
            >>> print(changes)
            ['learning_rate: 0.1 → 2e-05']
        """
        issues = self.validate()
        changes = []
        
        for issue in issues:
            if issue.auto_fixable and issue.param_name and issue.fixed_value is not None:
                old_value = getattr(self, issue.param_name)
                setattr(self, issue.param_name, issue.fixed_value)
                changes.append(f"{issue.param_name}: {old_value} → {issue.fixed_value}")
                logger.info(f"Auto-fixed: {issue.param_name} = {issue.fixed_value}")
        
        return changes


# =============================================================================
# SFT Preset - Supervised Fine-Tuning
# =============================================================================


@dataclass
class SFTPreset(BasePreset):
    """
    Preset for Supervised Fine-Tuning (SFT).
    
    Optimized for instruction-following and chat fine-tuning.
    Uses full fine-tuning of all model parameters.
    
    Recommended for:
    - Instruction tuning
    - Chat model training
    - Domain adaptation
    
    Example:
        >>> preset = SFTPreset(output_dir="./sft_model")
        >>> args = preset.get_training_args()
        >>> trainer = Trainer(model=model, args=args, ...)
    """
    
    # SFT typically uses smaller LR than pretraining
    learning_rate: float = field(default=2e-5)
    
    # Full fine-tuning usually needs gradient checkpointing
    gradient_checkpointing: bool = field(default=True)
    
    # NEFTune for better instruction following
    neftune_noise_alpha: Optional[float] = field(default=5.0)
    
    # Group by length for efficiency
    group_by_length: bool = field(default=True)
    
    def get_training_args(self) -> TrainingArguments:
        """Create SFT-optimized TrainingArguments."""
        base_args = super().get_training_args()
        
        # Add SFT-specific args
        if self.neftune_noise_alpha is not None:
            base_args.neftune_noise_alpha = self.neftune_noise_alpha
        base_args.group_by_length = self.group_by_length
        
        return base_args


# =============================================================================
# LoRA Preset - Low-Rank Adaptation
# =============================================================================


@dataclass
class LoRAPreset(BasePreset):
    """
    Preset for LoRA (Low-Rank Adaptation) fine-tuning.
    
    Uses PEFT library for parameter-efficient training.
    Trains only ~0.1-1% of parameters while achieving
    comparable results to full fine-tuning.
    
    Recommended for:
    - Limited GPU memory
    - Fast experimentation
    - Multiple task adapters
    
    Example:
        >>> preset = LoRAPreset(output_dir="./lora_model", lora_r=16)
        >>> training_args = preset.get_training_args()
        >>> lora_config = preset.get_lora_config()
        >>> 
        >>> from peft import get_peft_model
        >>> model = get_peft_model(model, lora_config)
        >>> trainer = Trainer(model=model, args=training_args, ...)
    """
    
    # LoRA hyperparameters
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    
    # Target modules (None = auto-detect)
    lora_target_modules: Optional[List[str]] = field(default=None)
    
    # LoRA uses higher LR
    learning_rate: float = field(default=2e-4)
    
    # Usually no need for gradient checkpointing with LoRA
    gradient_checkpointing: bool = field(default=False)
    
    # Bias training
    lora_bias: str = field(default="none")
    
    # Task type for PEFT
    task_type: str = field(default="CAUSAL_LM")
    
    def get_lora_config(self) -> Dict[str, Any]:
        """
        Get LoRA configuration dictionary for PEFT.
        
        Returns:
            Dict suitable for peft.LoraConfig
            
        Example:
            >>> from peft import LoraConfig
            >>> config = LoraConfig(**preset.get_lora_config())
        """
        config = {
            "r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "bias": self.lora_bias,
            "task_type": self.task_type,
        }
        
        if self.lora_target_modules:
            config["target_modules"] = self.lora_target_modules
        
        return config
    
    def summary(self) -> str:
        """Get summary including LoRA-specific settings."""
        base = super().summary()
        lora_info = [
            "",
            "LoRA Settings:",
            f"  r: {self.lora_r}",
            f"  alpha: {self.lora_alpha}",
            f"  dropout: {self.lora_dropout}",
            f"  targets: {self.lora_target_modules or 'auto'}",
        ]
        return base + "\n" + "\n".join(lora_info)


# =============================================================================
# QLoRA Preset - Quantized LoRA (4-bit)
# =============================================================================


@dataclass
class QLoRAPreset(LoRAPreset):
    """
    Preset for QLoRA (Quantized LoRA) fine-tuning.
    
    Combines 4-bit quantization with LoRA for extreme 
    memory efficiency. Can train 7B+ models on consumer GPUs.
    
    Recommended for:
    - Very limited GPU memory (8-16GB)
    - Training large models (7B, 13B, 70B)
    - Consumer hardware (RTX 3090, 4090)
    
    Requirements:
    - bitsandbytes library
    - CUDA-capable GPU
    
    Example:
        >>> preset = QLoRAPreset(output_dir="./qlora_model")
        >>> training_args = preset.get_training_args()
        >>> bnb_config = preset.get_bnb_config()
        >>> lora_config = preset.get_lora_config()
        >>> 
        >>> model = AutoModelForCausalLM.from_pretrained(
        ...     model_name,
        ...     quantization_config=bnb_config
        ... )
    """
    
    # QLoRA-specific quantization settings
    load_in_4bit: bool = field(default=True)
    bnb_4bit_quant_type: str = field(default="nf4")
    bnb_4bit_compute_dtype: str = field(default="bfloat16")
    bnb_4bit_use_double_quant: bool = field(default=True)
    
    # QLoRA typically uses higher rank
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    
    # Paged optimizer for memory efficiency
    optim: str = field(default="paged_adamw_8bit")
    
    # Enable gradient checkpointing for QLoRA
    gradient_checkpointing: bool = field(default=True)
    
    def get_bnb_config(self) -> Dict[str, Any]:
        """
        Get BitsAndBytes configuration for 4-bit quantization.
        
        Returns:
            Dict suitable for BitsAndBytesConfig
            
        Example:
            >>> from transformers import BitsAndBytesConfig
            >>> bnb_config = BitsAndBytesConfig(**preset.get_bnb_config())
        """
        compute_dtype_map = {
            "bfloat16": "torch.bfloat16",
            "float16": "torch.float16",
            "float32": "torch.float32",
        }
        
        return {
            "load_in_4bit": self.load_in_4bit,
            "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
            "bnb_4bit_compute_dtype": self.bnb_4bit_compute_dtype,
            "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
        }
    
    def get_training_args(self) -> TrainingArguments:
        """Create QLoRA-optimized TrainingArguments."""
        base_args = super().get_training_args()
        base_args.optim = self.optim
        return base_args
    
    def summary(self) -> str:
        """Get summary including QLoRA-specific settings."""
        base = super().summary()
        qlora_info = [
            "",
            "QLoRA Quantization:",
            f"  4-bit: {self.load_in_4bit}",
            f"  quant_type: {self.bnb_4bit_quant_type}",
            f"  compute_dtype: {self.bnb_4bit_compute_dtype}",
            f"  double_quant: {self.bnb_4bit_use_double_quant}",
            f"  optimizer: {self.optim}",
        ]
        return base + "\n" + "\n".join(qlora_info)


# =============================================================================
# DPO Preset - Direct Preference Optimization
# =============================================================================


@dataclass
class DPOPreset(BasePreset):
    """
    Preset for DPO (Direct Preference Optimization) training.
    
    Optimized for RLHF-style preference learning without
    requiring a separate reward model.
    
    Recommended for:
    - Preference-based fine-tuning
    - Alignment training
    - RLHF without reward model
    
    Requires:
    - TRL library (trl)
    - Preference dataset with chosen/rejected pairs
    
    Example:
        >>> from trl import DPOTrainer
        >>> preset = DPOPreset(output_dir="./dpo_model")
        >>> training_args = preset.get_training_args()
        >>> 
        >>> trainer = DPOTrainer(
        ...     model=model,
        ...     args=training_args,
        ...     beta=preset.dpo_beta,
        ...     ...
        ... )
    """
    
    # DPO-specific hyperparameters
    dpo_beta: float = field(default=0.1)
    dpo_loss_type: str = field(default="sigmoid")
    
    # Reference model settings
    ref_model_mixup_alpha: float = field(default=0.0)
    precompute_ref_log_probs: bool = field(default=False)
    
    # DPO typically uses lower LR
    learning_rate: float = field(default=5e-7)
    
    # Longer warmup for stability
    warmup_ratio: float = field(default=0.1)
    
    # Gradient checkpointing for memory
    gradient_checkpointing: bool = field(default=True)
    
    # Smaller batch size (DPO processes pairs)
    per_device_train_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=8)
    
    def get_dpo_config(self) -> Dict[str, Any]:
        """
        Get DPO-specific configuration for TRL.
        
        Returns:
            Dict with DPO hyperparameters
        """
        return {
            "beta": self.dpo_beta,
            "loss_type": self.dpo_loss_type,
            "ref_model_mixup_alpha": self.ref_model_mixup_alpha,
            "precompute_ref_log_probs": self.precompute_ref_log_probs,
        }
    
    def summary(self) -> str:
        """Get summary including DPO-specific settings."""
        base = super().summary()
        dpo_info = [
            "",
            "DPO Settings:",
            f"  beta: {self.dpo_beta}",
            f"  loss_type: {self.dpo_loss_type}",
        ]
        return base + "\n" + "\n".join(dpo_info)


# =============================================================================
# Memory Efficient Preset
# =============================================================================


@dataclass
class MemoryEfficientPreset(BasePreset):
    """
    Preset optimized for limited GPU memory.
    
    Uses all available memory optimization techniques:
    - Gradient checkpointing
    - Small batch with accumulation
    - 8-bit optimizer
    - Model offloading options
    
    Recommended for:
    - Consumer GPUs (8-24GB)
    - Training larger models than usually possible
    - Preventing OOM errors
    
    Example:
        >>> preset = MemoryEfficientPreset(
        ...     output_dir="./model",
        ...     target_gpu_memory_gb=16
        ... )
        >>> args = preset.get_training_args()
    """
    
    # Target GPU memory (for auto-tuning)
    target_gpu_memory_gb: int = field(default=16)
    
    # Memory optimization flags
    gradient_checkpointing: bool = field(default=True)
    
    # Small batch, high accumulation
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=16)
    
    # 8-bit optimizer
    optim: str = field(default="adamw_8bit")
    
    # bf16 for memory efficiency
    bf16: bool = field(default=True)
    
    # Empty cache periodically
    torch_empty_cache_steps: int = field(default=100)
    
    # Auto batch size finder
    auto_find_batch_size: bool = field(default=False)
    
    def get_training_args(self) -> TrainingArguments:
        """Create memory-optimized TrainingArguments."""
        args = super().get_training_args()
        args.optim = self.optim
        
        if self.torch_empty_cache_steps:
            args.torch_empty_cache_steps = self.torch_empty_cache_steps
            
        args.auto_find_batch_size = self.auto_find_batch_size
        
        return args
    
    def get_memory_tips(self) -> List[str]:
        """Get additional memory optimization tips."""
        return [
            "💡 Memory Optimization Tips:",
            "",
            "1. Use gradient_checkpointing=True (already enabled)",
            "2. Reduce max_length in tokenizer if possible",
            "3. Use torch.compile() for PyTorch 2.0+ (may reduce memory)",
            "4. Consider using LoRA/QLoRA for large models",
            "5. Use DeepSpeed ZeRO-3 for multi-GPU",
            "6. Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            "",
            f"Target GPU: {self.target_gpu_memory_gb}GB",
            f"Effective batch: {self.get_effective_batch_size()}",
        ]
    
    def summary(self) -> str:
        """Get summary with memory tips."""
        base = super().summary()
        memory_info = [
            "",
            "Memory Optimization:",
            f"  Target GPU: {self.target_gpu_memory_gb}GB",
            f"  Optimizer: {self.optim}",
            f"  Empty cache: every {self.torch_empty_cache_steps} steps",
            f"  Auto batch size: {self.auto_find_batch_size}",
        ]
        return base + "\n" + "\n".join(memory_info)


# =============================================================================
# Preset Registry and Factory
# =============================================================================


# Registry of available presets
_PRESET_REGISTRY: Dict[str, type] = {
    "sft": SFTPreset,
    "lora": LoRAPreset,
    "qlora": QLoRAPreset,
    "dpo": DPOPreset,
    "memory_efficient": MemoryEfficientPreset,
    "memory": MemoryEfficientPreset,  # alias
}


def list_presets() -> List[str]:
    """
    List all available preset names.
    
    Returns:
        List of preset names that can be used with get_preset()
        
    Example:
        >>> from transformers.training_presets import list_presets
        >>> print(list_presets())
        ['sft', 'lora', 'qlora', 'dpo', 'memory_efficient', 'memory']
    """
    return list(_PRESET_REGISTRY.keys())


def get_preset(
    name: str,
    **kwargs
) -> BasePreset:
    """
    Get a training preset by name.
    
    Args:
        name: Preset name (see list_presets() for options)
        **kwargs: Override any preset parameter
        
    Returns:
        Configured preset instance
        
    Raises:
        ValueError: If preset name is unknown
        
    Example:
        >>> preset = get_preset("lora", learning_rate=1e-4)
        >>> args = preset.get_training_args()
    """
    name_lower = name.lower()
    
    if name_lower not in _PRESET_REGISTRY:
        available = ", ".join(list_presets())
        raise ValueError(
            f"Unknown preset: '{name}'. Available: {available}"
        )
    
    preset_class = _PRESET_REGISTRY[name_lower]
    return preset_class(**kwargs)


def register_preset(name: str, preset_class: type):
    """
    Register a custom preset.
    
    Args:
        name: Name for the preset
        preset_class: Class that inherits from BasePreset
        
    Example:
        >>> @dataclass
        >>> class MyCustomPreset(BasePreset):
        ...     my_param: int = 42
        >>> 
        >>> register_preset("custom", MyCustomPreset)
        >>> preset = get_preset("custom")
    """
    if not issubclass(preset_class, BasePreset):
        raise TypeError(
            f"Preset class must inherit from BasePreset, got {preset_class}"
        )
    _PRESET_REGISTRY[name.lower()] = preset_class
    logger.info(f"Registered custom preset: {name}")


# =============================================================================
# Convenience Functions
# =============================================================================


def quick_sft_args(output_dir: str = "./sft_output", **kwargs) -> TrainingArguments:
    """
    Quick way to get SFT TrainingArguments.
    
    Args:
        output_dir: Output directory
        **kwargs: Override any SFT preset parameter
        
    Returns:
        TrainingArguments ready for SFT
        
    Example:
        >>> args = quick_sft_args("./my_model", learning_rate=1e-5)
        >>> trainer = Trainer(model=model, args=args, ...)
    """
    return SFTPreset(output_dir=output_dir, **kwargs).get_training_args()


def quick_lora_args(output_dir: str = "./lora_output", **kwargs) -> TrainingArguments:
    """
    Quick way to get LoRA TrainingArguments.
    
    Args:
        output_dir: Output directory
        **kwargs: Override any LoRA preset parameter
        
    Returns:
        TrainingArguments ready for LoRA training
    """
    return LoRAPreset(output_dir=output_dir, **kwargs).get_training_args()


def quick_qlora_args(output_dir: str = "./qlora_output", **kwargs) -> TrainingArguments:
    """
    Quick way to get QLoRA TrainingArguments.
    
    Args:
        output_dir: Output directory
        **kwargs: Override any QLoRA preset parameter
        
    Returns:
        TrainingArguments ready for QLoRA training
    """
    return QLoRAPreset(output_dir=output_dir, **kwargs).get_training_args()


# =============================================================================
# Auto-detection utilities
# =============================================================================


def suggest_preset_for_model(
    model_name_or_path: str,
    available_gpu_memory_gb: float = 24.0,
    task: str = "sft"
) -> str:
    """
    Suggest the best preset based on model and available resources.
    
    Args:
        model_name_or_path: HuggingFace model name or path
        available_gpu_memory_gb: Available GPU memory in GB
        task: Training task ("sft", "dpo", "chat")
        
    Returns:
        Recommended preset name
        
    Example:
        >>> preset_name = suggest_preset_for_model(
        ...     "meta-llama/Llama-2-7b",
        ...     available_gpu_memory_gb=16
        ... )
        >>> print(preset_name)  # 'qlora'
    """
    # Rough model size estimation from name
    model_lower = model_name_or_path.lower()
    
    estimated_params_b = 0.0
    if "70b" in model_lower:
        estimated_params_b = 70.0
    elif "34b" in model_lower or "33b" in model_lower:
        estimated_params_b = 34.0
    elif "13b" in model_lower or "14b" in model_lower:
        estimated_params_b = 13.0
    elif "7b" in model_lower or "8b" in model_lower:
        estimated_params_b = 7.0
    elif "3b" in model_lower:
        estimated_params_b = 3.0
    elif "1b" in model_lower or "1.5b" in model_lower:
        estimated_params_b = 1.5
    else:
        estimated_params_b = 1.0  # assume small
    
    # Rough memory requirement: ~2x params for bf16 training
    required_memory = estimated_params_b * 2
    
    if task.lower() == "dpo":
        return "dpo"
    
    if required_memory > available_gpu_memory_gb:
        # Need memory-efficient approach
        if required_memory > available_gpu_memory_gb * 4:
            return "qlora"  # Very constrained
        else:
            return "lora"
    
    return "sft"


def print_preset_comparison():
    """Print a comparison table of all presets."""
    print("\n" + "=" * 70)
    print("TRAINING PRESETS COMPARISON")
    print("=" * 70)
    print(f"{'Preset':<15} {'Memory':<12} {'Speed':<10} {'Quality':<10} {'Use Case':<20}")
    print("-" * 70)
    print(f"{'SFT':<15} {'High':<12} {'Medium':<10} {'Best':<10} {'Full fine-tuning':<20}")
    print(f"{'LoRA':<15} {'Medium':<12} {'Fast':<10} {'Good':<10} {'Efficient tuning':<20}")
    print(f"{'QLoRA':<15} {'Low':<12} {'Medium':<10} {'Good':<10} {'Large models':<20}")
    print(f"{'DPO':<15} {'High':<12} {'Slow':<10} {'Best':<10} {'Alignment':<20}")
    print(f"{'Memory':<15} {'Lowest':<12} {'Slower':<10} {'Good':<10} {'Limited GPU':<20}")
    print("=" * 70 + "\n")


# =============================================================================
# Interactive Configuration Validation
# =============================================================================


def validate_config(preset: BasePreset, interactive: bool = True) -> bool:
    """
    Интерактивная валидация конфигурации с диалогом.
    
    Проверяет конфигурацию на проблемы и предлагает пользователю
    выбрать действие:
    - [Y] Продолжить на свой риск
    - [N] Остановить обучение
    - [A] Автоматически исправить
    
    Args:
        preset: Preset для валидации
        interactive: Если True, показывает интерактивный диалог
                    Если False, только логирует warnings
                    
    Returns:
        True если можно продолжить обучение, False если отменено
        
    Example:
        >>> from transformers.training_presets import get_preset, validate_config
        >>> 
        >>> preset = get_preset("qlora", output_dir="./model")
        >>> 
        >>> if validate_config(preset, interactive=True):
        ...     args = preset.get_training_args()
        ...     trainer.train()
        ... else:
        ...     print("Обучение отменено")
    """
    issues = preset.validate()
    
    # Нет проблем
    if not issues:
        print()
        print("=" * 60)
        print("✅ КОНФИГУРАЦИЯ КОРРЕКТНА")
        print("=" * 60)
        print()
        return True
    
    # Показываем проблемы
    print()
    print("=" * 60)
    print("⚠️  ОБНАРУЖЕНЫ ПРОБЛЕМЫ В КОНФИГУРАЦИИ")
    print("=" * 60)
    
    for issue in issues:
        icon = "⚠️" if issue.level == "warning" else "❌"
        fixable = "[auto-fix]" if issue.auto_fixable else "[manual]"
        print(f"  {icon} {issue.message}")
        print(f"     💡 {issue.fix_suggestion} {fixable}")
    
    print("=" * 60)
    
    # Неинтерактивный режим
    if not interactive:
        print("⚠️  Режим без интерактива. Продолжаем с текущими настройками...")
        logger.warning(f"Найдено {len(issues)} проблем в конфигурации, продолжаем без исправлений")
        return True
    
    # Интерактивный диалог
    print()
    print("Выберите действие:")
    print("  [Y] Продолжить на свой риск")
    print("  [N] Остановить обучение")
    print("  [A] Автоматически исправить")
    print()
    
    while True:
        try:
            choice = input("Ваш выбор [Y/N/A]: ").strip().upper()
        except EOFError:
            # Неинтерактивный режим (pipe, CI/CD)
            print("⚠️  Нет ввода (EOF). Продолжаем с текущими настройками...")
            return True
        except KeyboardInterrupt:
            print("\n❌ Прервано пользователем.")
            return False
        
        if choice == 'Y':
            print()
            print("⚠️  Продолжаем на свой риск...")
            return True
            
        elif choice == 'N':
            print()
            print("❌ Обучение отменено пользователем.")
            return False
            
        elif choice == 'A':
            changes = preset.auto_fix()
            print()
            if changes:
                print("✅ Автоматическое исправление:")
                for change in changes:
                    print(f"   • {change}")
            else:
                print("ℹ️  Нет автоматически исправляемых проблем.")
            print()
            return True
            
        else:
            print("❓ Неизвестный выбор. Введите Y, N или A.")
