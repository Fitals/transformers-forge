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
Dataset Utilities for Transformers Forge.

Утилиты для анализа и работы с датасетами перед обучением.

Key features:
- Анализ структуры датасета
- Подсчёт токенов
- Рекомендации по batch size
- Оценка времени обучения

Usage:
    from transformers.dataset_utils import (
        analyze_dataset,
        estimate_tokens,
        recommend_batch_size,
        DatasetAnalyzer
    )
    
    # Quick analysis
    stats = analyze_dataset(dataset, tokenizer)
    print(f"Total tokens: {stats['total_tokens']:,}")
    
    # Full analyzer
    analyzer = DatasetAnalyzer(dataset, tokenizer)
    analyzer.print_report()

Added in v1.1.4.
"""

import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from .utils import logging


logger = logging.get_logger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class DatasetStats:
    """
    Statistics about a dataset.
    
    Содержит полную статистику о датасете после анализа.
    """
    # Basic counts
    num_samples: int = 0
    num_columns: int = 0
    column_names: List[str] = field(default_factory=list)
    
    # Token statistics
    total_tokens: int = 0
    min_tokens: int = 0
    max_tokens: int = 0
    mean_tokens: float = 0.0
    median_tokens: float = 0.0
    std_tokens: float = 0.0
    
    # Length distribution buckets
    length_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Memory estimates
    estimated_memory_mb: float = 0.0
    
    # Recommendations
    recommended_batch_size: int = 4
    recommended_max_length: int = 512
    estimated_training_time_minutes: float = 0.0
    
    def __repr__(self) -> str:
        return (
            f"DatasetStats(samples={self.num_samples}, "
            f"tokens={self.total_tokens:,}, "
            f"mean_len={self.mean_tokens:.1f})"
        )


# =============================================================================
# Core Analysis Functions
# =============================================================================


def analyze_dataset(
    dataset: Any,
    tokenizer: Optional[Any] = None,
    text_column: Optional[str] = None,
    max_samples: int = 1000,
    verbose: bool = False,
) -> DatasetStats:
    """
    Analyze a dataset and return comprehensive statistics.
    
    Быстрый анализ датасета с подсчётом токенов и статистики длин.
    
    Args:
        dataset: HuggingFace Dataset or list of dicts
        tokenizer: Optional tokenizer for token counting
        text_column: Column containing text (auto-detected if None)
        max_samples: Maximum samples to analyze (for speed)
        verbose: Print progress
        
    Returns:
        DatasetStats with comprehensive statistics
        
    Example:
        >>> from datasets import load_dataset
        >>> from transformers import AutoTokenizer
        >>> 
        >>> dataset = load_dataset("json", data_files="data.json", split="train")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        >>> 
        >>> stats = analyze_dataset(dataset, tokenizer)
        >>> print(f"Dataset has {stats.num_samples} samples")
        >>> print(f"Average length: {stats.mean_tokens:.1f} tokens")
    """
    stats = DatasetStats()
    
    # Handle different dataset types
    if hasattr(dataset, '__len__'):
        stats.num_samples = len(dataset)
    else:
        # Try to iterate and count
        try:
            stats.num_samples = sum(1 for _ in dataset)
        except:
            stats.num_samples = 0
    
    # Get column names
    if hasattr(dataset, 'column_names'):
        stats.column_names = dataset.column_names
        stats.num_columns = len(stats.column_names)
    elif stats.num_samples > 0:
        try:
            first_item = dataset[0]
            if isinstance(first_item, dict):
                stats.column_names = list(first_item.keys())
                stats.num_columns = len(stats.column_names)
        except:
            pass
    
    # Auto-detect text column
    if text_column is None:
        text_column = _detect_text_column(stats.column_names)
    
    if verbose:
        print(f"Analyzing dataset: {stats.num_samples} samples")
        print(f"Text column: {text_column}")
    
    # Sample for analysis
    sample_size = min(max_samples, stats.num_samples)
    
    # Collect token lengths
    token_lengths = []
    
    if tokenizer is not None and text_column is not None:
        if verbose:
            print(f"Tokenizing {sample_size} samples...")
        
        for i in range(sample_size):
            try:
                item = dataset[i]
                text = _extract_text(item, text_column)
                
                if text:
                    tokens = tokenizer(text, truncation=False, add_special_tokens=True)
                    token_lengths.append(len(tokens["input_ids"]))
            except Exception as e:
                if verbose:
                    logger.warning(f"Error tokenizing sample {i}: {e}")
                continue
    elif text_column is not None:
        # Fallback: estimate tokens as words * 1.3
        for i in range(sample_size):
            try:
                item = dataset[i]
                text = _extract_text(item, text_column)
                
                if text:
                    word_count = len(text.split())
                    estimated_tokens = int(word_count * 1.3)
                    token_lengths.append(estimated_tokens)
            except:
                continue
    
    # Calculate statistics
    if token_lengths:
        stats.total_tokens = int(sum(token_lengths) * (stats.num_samples / sample_size))
        stats.min_tokens = min(token_lengths)
        stats.max_tokens = max(token_lengths)
        stats.mean_tokens = statistics.mean(token_lengths)
        stats.median_tokens = statistics.median(token_lengths)
        
        if len(token_lengths) > 1:
            stats.std_tokens = statistics.stdev(token_lengths)
        
        # Length distribution
        stats.length_distribution = _calculate_length_distribution(token_lengths)
        
        # Recommendations
        stats.recommended_max_length = _recommend_max_length(stats.max_tokens, stats.mean_tokens)
        stats.recommended_batch_size = _recommend_batch_size_from_stats(stats.mean_tokens)
        
        # Memory estimate (rough: 4 bytes per token for fp32)
        stats.estimated_memory_mb = (stats.total_tokens * 4) / (1024 * 1024)
    
    if verbose:
        print(f"Analysis complete: {stats.total_tokens:,} total tokens")
    
    return stats


def _detect_text_column(columns: List[str]) -> Optional[str]:
    """Auto-detect the text column from common names."""
    priority_columns = [
        "text",
        "content", 
        "messages",
        "instruction",
        "input",
        "prompt",
        "question",
        "context",
    ]
    
    for col in priority_columns:
        if col in columns:
            return col
    
    # Return first column as fallback
    return columns[0] if columns else None


def _extract_text(item: Any, text_column: str) -> Optional[str]:
    """Extract text from a dataset item."""
    if isinstance(item, dict):
        value = item.get(text_column)
        
        # Handle messages format (chat)
        if text_column == "messages" and isinstance(value, list):
            texts = []
            for msg in value:
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                    texts.append(str(content))
            return " ".join(texts)
        
        if isinstance(value, str):
            return value
        elif value is not None:
            return str(value)
    
    return None


def _calculate_length_distribution(lengths: List[int]) -> Dict[str, int]:
    """Calculate distribution of token lengths."""
    distribution = {
        "0-128": 0,
        "128-256": 0,
        "256-512": 0,
        "512-1024": 0,
        "1024-2048": 0,
        "2048-4096": 0,
        "4096+": 0,
    }
    
    for length in lengths:
        if length <= 128:
            distribution["0-128"] += 1
        elif length <= 256:
            distribution["128-256"] += 1
        elif length <= 512:
            distribution["256-512"] += 1
        elif length <= 1024:
            distribution["512-1024"] += 1
        elif length <= 2048:
            distribution["1024-2048"] += 1
        elif length <= 4096:
            distribution["2048-4096"] += 1
        else:
            distribution["4096+"] += 1
    
    return distribution


def _recommend_max_length(max_tokens: int, mean_tokens: float) -> int:
    """Recommend max_length based on distribution."""
    # Use 95th percentile approximation: mean + 2*std roughly
    recommended = int(mean_tokens * 2)
    
    # Round up to power of 2 or common values
    common_lengths = [128, 256, 512, 1024, 2048, 4096, 8192]
    
    for length in common_lengths:
        if recommended <= length:
            return length
    
    return 8192


def _recommend_batch_size_from_stats(mean_tokens: float) -> int:
    """Recommend batch size based on sequence length."""
    # Longer sequences = smaller batch size
    if mean_tokens <= 128:
        return 16
    elif mean_tokens <= 256:
        return 8
    elif mean_tokens <= 512:
        return 4
    elif mean_tokens <= 1024:
        return 2
    else:
        return 1


# =============================================================================
# Token Estimation
# =============================================================================


def estimate_tokens(
    dataset: Any,
    tokenizer: Optional[Any] = None,
    text_column: Optional[str] = None,
    sample_size: int = 100,
) -> int:
    """
    Quickly estimate total tokens in a dataset.
    
    Быстрая оценка общего количества токенов в датасете.
    Использует выборку для скорости.
    
    Args:
        dataset: HuggingFace Dataset or list
        tokenizer: Tokenizer for accurate counting
        text_column: Column with text
        sample_size: Samples for estimation
        
    Returns:
        Estimated total token count
        
    Example:
        >>> tokens = estimate_tokens(dataset, tokenizer)
        >>> print(f"~{tokens:,} tokens in dataset")
    """
    stats = analyze_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        text_column=text_column,
        max_samples=sample_size,
        verbose=False,
    )
    
    return stats.total_tokens


def estimate_training_time(
    stats: DatasetStats,
    num_epochs: int = 3,
    batch_size: int = 4,
    tokens_per_second: float = 1000.0,
) -> Dict[str, Any]:
    """
    Estimate training time based on dataset statistics.
    
    Args:
        stats: DatasetStats from analyze_dataset
        num_epochs: Number of training epochs
        batch_size: Training batch size
        tokens_per_second: Estimated training throughput
        
    Returns:
        Dict with time estimates
        
    Example:
        >>> time_info = estimate_training_time(stats, num_epochs=3)
        >>> print(f"Estimated: {time_info['formatted']}")
    """
    total_tokens = stats.total_tokens * num_epochs
    
    total_seconds = total_tokens / tokens_per_second
    
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    
    return {
        "total_tokens": total_tokens,
        "total_seconds": total_seconds,
        "hours": hours,
        "minutes": minutes,
        "seconds": seconds,
        "formatted": f"{hours}h {minutes}m {seconds}s",
        "tokens_per_second": tokens_per_second,
    }


# =============================================================================
# Batch Size Recommendations
# =============================================================================


def recommend_batch_size(
    model: Optional[Any] = None,
    dataset_stats: Optional[DatasetStats] = None,
    max_length: int = 512,
    available_memory_gb: Optional[float] = None,
    dtype: str = "bf16",
) -> Dict[str, Any]:
    """
    Recommend optimal batch size based on model, data, and hardware.
    
    Анализирует модель, датасет и доступную память для рекомендации
    оптимального batch size.
    
    Args:
        model: PyTorch model (optional, for parameter count)
        dataset_stats: Statistics from analyze_dataset
        max_length: Maximum sequence length
        available_memory_gb: GPU memory available
        dtype: Training dtype ("fp32", "fp16", "bf16")
        
    Returns:
        Dict with recommendations:
            - batch_size: Recommended batch size
            - gradient_accumulation: Recommended accumulation steps
            - effective_batch: Effective batch size
            - reason: Explanation
            
    Example:
        >>> rec = recommend_batch_size(model, stats, available_memory_gb=16.0)
        >>> print(f"Use batch_size={rec['batch_size']}")
    """
    result = {
        "batch_size": 4,
        "gradient_accumulation": 1,
        "effective_batch": 4,
        "reason": [],
    }
    
    # Base recommendation from sequence length
    if dataset_stats is not None:
        mean_len = dataset_stats.mean_tokens
        if mean_len > 2048:
            result["batch_size"] = 1
            result["reason"].append(f"Long sequences ({mean_len:.0f} tokens avg)")
        elif mean_len > 1024:
            result["batch_size"] = 2
            result["reason"].append(f"Medium-long sequences ({mean_len:.0f} tokens avg)")
        elif mean_len > 512:
            result["batch_size"] = 4
        else:
            result["batch_size"] = 8
            result["reason"].append(f"Short sequences ({mean_len:.0f} tokens avg)")
    
    # Adjust for model size
    if model is not None:
        try:
            total_params = sum(p.numel() for p in model.parameters())
            params_b = total_params / 1e9
            
            if params_b > 7:
                result["batch_size"] = max(1, result["batch_size"] // 4)
                result["reason"].append(f"Large model ({params_b:.1f}B params)")
            elif params_b > 3:
                result["batch_size"] = max(1, result["batch_size"] // 2)
                result["reason"].append(f"Medium model ({params_b:.1f}B params)")
        except:
            pass
    
    # Adjust for available memory
    if available_memory_gb is not None:
        if available_memory_gb < 8:
            result["batch_size"] = max(1, result["batch_size"] // 2)
            result["reason"].append(f"Limited GPU memory ({available_memory_gb:.0f}GB)")
        elif available_memory_gb >= 24:
            result["batch_size"] = min(16, result["batch_size"] * 2)
            result["reason"].append(f"Large GPU memory ({available_memory_gb:.0f}GB)")
    
    # Calculate gradient accumulation to reach effective batch of 16-32
    target_effective = 16
    result["gradient_accumulation"] = max(1, target_effective // result["batch_size"])
    result["effective_batch"] = result["batch_size"] * result["gradient_accumulation"]
    
    if not result["reason"]:
        result["reason"].append("Default balanced configuration")
    
    result["reason"] = "; ".join(result["reason"])
    
    return result


# =============================================================================
# Dataset Analyzer Class
# =============================================================================


class DatasetAnalyzer:
    """
    Comprehensive dataset analyzer with detailed reports.
    
    Комплексный анализатор датасетов с детальными отчётами
    и рекомендациями для обучения.
    
    Example:
        >>> analyzer = DatasetAnalyzer(dataset, tokenizer)
        >>> analyzer.analyze()
        >>> analyzer.print_report()
        >>> 
        >>> # Get recommendations
        >>> rec = analyzer.get_recommendations(model)
        >>> print(f"Recommended batch size: {rec['batch_size']}")
    """
    
    def __init__(
        self,
        dataset: Any,
        tokenizer: Optional[Any] = None,
        text_column: Optional[str] = None,
    ):
        """
        Initialize analyzer.
        
        Args:
            dataset: HuggingFace Dataset or list of dicts
            tokenizer: Optional tokenizer for accurate token counts
            text_column: Column containing text (auto-detected if None)
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.stats: Optional[DatasetStats] = None
    
    def analyze(self, max_samples: int = 1000, verbose: bool = False) -> DatasetStats:
        """
        Run comprehensive analysis.
        
        Args:
            max_samples: Maximum samples to analyze
            verbose: Print progress
            
        Returns:
            DatasetStats with comprehensive statistics
        """
        self.stats = analyze_dataset(
            dataset=self.dataset,
            tokenizer=self.tokenizer,
            text_column=self.text_column,
            max_samples=max_samples,
            verbose=verbose,
        )
        return self.stats
    
    def get_recommendations(
        self,
        model: Optional[Any] = None,
        available_memory_gb: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Get training recommendations based on analysis.
        
        Args:
            model: PyTorch model for size-aware recommendations
            available_memory_gb: Available GPU memory
            
        Returns:
            Dict with all recommendations
        """
        if self.stats is None:
            self.analyze()
        
        batch_rec = recommend_batch_size(
            model=model,
            dataset_stats=self.stats,
            available_memory_gb=available_memory_gb,
        )
        
        return {
            "batch_size": batch_rec["batch_size"],
            "gradient_accumulation": batch_rec["gradient_accumulation"],
            "effective_batch_size": batch_rec["effective_batch"],
            "max_length": self.stats.recommended_max_length,
            "num_samples": self.stats.num_samples,
            "total_tokens": self.stats.total_tokens,
            "reason": batch_rec["reason"],
        }
    
    def print_report(self):
        """Print a formatted analysis report."""
        if self.stats is None:
            self.analyze()
        
        s = self.stats
        
        print()
        print("=" * 60)
        print("📊 DATASET ANALYSIS REPORT")
        print("=" * 60)
        
        print(f"\n📁 Basic Info:")
        print(f"   Samples:     {s.num_samples:,}")
        print(f"   Columns:     {s.num_columns}")
        if s.column_names:
            print(f"   Fields:      {', '.join(s.column_names[:5])}")
            if len(s.column_names) > 5:
                print(f"                ... and {len(s.column_names) - 5} more")
        
        print(f"\n📏 Token Statistics:")
        print(f"   Total:       {s.total_tokens:,}")
        print(f"   Min:         {s.min_tokens:,}")
        print(f"   Max:         {s.max_tokens:,}")
        print(f"   Mean:        {s.mean_tokens:.1f}")
        print(f"   Median:      {s.median_tokens:.1f}")
        print(f"   Std Dev:     {s.std_tokens:.1f}")
        
        print(f"\n📈 Length Distribution:")
        total_samples = sum(s.length_distribution.values())
        for bucket, count in s.length_distribution.items():
            if count > 0:
                pct = (count / total_samples) * 100 if total_samples > 0 else 0
                bar = "█" * int(pct / 5)
                print(f"   {bucket:>10}: {count:>5} ({pct:>5.1f}%) {bar}")
        
        print(f"\n💡 Recommendations:")
        print(f"   Batch Size:  {s.recommended_batch_size}")
        print(f"   Max Length:  {s.recommended_max_length}")
        
        print("=" * 60)
        print()


# =============================================================================
# Convenience Functions
# =============================================================================


def quick_dataset_check(
    path: str,
    tokenizer: Optional[Any] = None,
    split: str = "train",
) -> DatasetStats:
    """
    Quick check of a dataset file.
    
    Быстрая проверка датасета из файла.
    
    Args:
        path: Path to dataset file (json, jsonl, csv, parquet)
        tokenizer: Optional tokenizer
        split: Dataset split to analyze
        
    Returns:
        DatasetStats
        
    Example:
        >>> stats = quick_dataset_check("data.json", tokenizer)
        >>> print(stats)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")
    
    # Detect format
    if path.endswith(".json") or path.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=path, split=split)
    elif path.endswith(".csv"):
        dataset = load_dataset("csv", data_files=path, split=split)
    elif path.endswith(".parquet"):
        dataset = load_dataset("parquet", data_files=path, split=split)
    else:
        # Try json as default
        dataset = load_dataset("json", data_files=path, split=split)
    
    return analyze_dataset(dataset, tokenizer)
