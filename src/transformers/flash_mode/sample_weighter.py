# Copyright 2024 Transformers Forge Contributors.
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
Sample Weighter — Адаптивное взвешивание примеров
=================================================

Снижает вес примеров с низким loss (уже "выученных"),
позволяя модели фокусироваться на сложных примерах.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Any

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class SampleWeightingStats:
    """Статистика взвешивания примеров."""
    total_samples: int = 0
    low_loss_samples: int = 0
    high_loss_samples: int = 0
    average_weight: float = 1.0


class SampleWeighter:
    """
    Адаптивное взвешивание примеров по loss.
    
    Принцип работы:
    - Примеры с низким loss → сниженный вес (минимум 0.3)
    - Примеры с высоким loss → полный вес (1.0)
    - Плавный переход между ними
    
    Args:
        min_weight: Минимальный вес примера (default: 0.3)
        low_loss_threshold: Порог для "выученного" примера
        high_loss_threshold: Порог для "сложного" примера
        enable_stats: Собирать статистику
    """
    
    def __init__(
        self,
        min_weight: float = 0.3,
        low_loss_threshold: float = 0.5,
        high_loss_threshold: Optional[float] = None,
        enable_stats: bool = True,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("SampleWeighter требует PyTorch")
        
        self.min_weight = min_weight
        self.low_loss_threshold = low_loss_threshold
        self.high_loss_threshold = high_loss_threshold or low_loss_threshold * 2.0
        self.enable_stats = enable_stats
        
        # Статистика
        self.stats = SampleWeightingStats()
        
        # История для адаптации порогов
        self._loss_history = []
        self._history_size = 100
    
    def compute_weights(
        self,
        losses: Any,  # torch.Tensor
    ) -> Any:
        """
        Вычисляет веса для каждого примера на основе loss.
        
        Args:
            losses: Тензор loss для каждого примера [batch_size]
            
        Returns:
            Тензор весов [batch_size]
        """
        if not TORCH_AVAILABLE:
            return losses
        
        # Нормализуем loss к [0, 1] диапазону
        # low_loss → weight близко к min_weight
        # high_loss → weight близко к 1.0
        
        # Sigmoid-подобное взвешивание
        # weight = min_weight + (1 - min_weight) * sigmoid((loss - mid) / scale)
        
        mid = (self.low_loss_threshold + self.high_loss_threshold) / 2
        scale = (self.high_loss_threshold - self.low_loss_threshold) / 4
        
        normalized = (losses - mid) / (scale + 1e-8)
        sigmoid_weights = torch.sigmoid(normalized)
        
        # Масштабируем к [min_weight, 1.0]
        weights = self.min_weight + (1.0 - self.min_weight) * sigmoid_weights
        
        # Обновляем статистику
        if self.enable_stats:
            self._update_stats(losses, weights)
        
        return weights
    
    def apply_weights(
        self,
        loss: Any,  # torch.Tensor скаляр или per-sample
        per_sample_losses: Optional[Any] = None,
    ) -> Any:
        """
        Применяет взвешивание к loss.
        
        Args:
            loss: Общий loss (скаляр)
            per_sample_losses: Опционально, loss для каждого примера
            
        Returns:
            Взвешенный loss
        """
        if per_sample_losses is None:
            # Нет per-sample losses — возвращаем как есть
            return loss
        
        weights = self.compute_weights(per_sample_losses)
        
        # Взвешенное среднее
        weighted_loss = (per_sample_losses * weights).mean()
        
        return weighted_loss
    
    def _update_stats(self, losses: Any, weights: Any):
        """Обновляет статистику."""
        batch_size = losses.numel()
        
        self.stats.total_samples += batch_size
        self.stats.low_loss_samples += (losses < self.low_loss_threshold).sum().item()
        self.stats.high_loss_samples += (losses > self.high_loss_threshold).sum().item()
        
        # Скользящее среднее веса
        avg_w = weights.mean().item()
        alpha = 0.1
        self.stats.average_weight = alpha * avg_w + (1 - alpha) * self.stats.average_weight
        
        # История для адаптации
        self._loss_history.append(losses.mean().item())
        if len(self._loss_history) > self._history_size:
            self._loss_history.pop(0)
    
    def adapt_thresholds(self):
        """
        Адаптирует пороги на основе истории loss.
        
        Вызывайте периодически для автоматической настройки.
        """
        if len(self._loss_history) < 20:
            return
        
        # Используем квантили для определения порогов
        sorted_losses = sorted(self._loss_history)
        
        # low_threshold = 25-й перцентиль
        low_idx = len(sorted_losses) // 4
        self.low_loss_threshold = sorted_losses[low_idx]
        
        # high_threshold = 75-й перцентиль
        high_idx = 3 * len(sorted_losses) // 4
        self.high_loss_threshold = sorted_losses[high_idx]
    
    def get_stats(self) -> dict:
        """Возвращает статистику."""
        total = max(1, self.stats.total_samples)
        
        return {
            "total_samples": self.stats.total_samples,
            "low_loss_rate": self.stats.low_loss_samples / total,
            "high_loss_rate": self.stats.high_loss_samples / total,
            "average_weight": self.stats.average_weight,
            "low_threshold": self.low_loss_threshold,
            "high_threshold": self.high_loss_threshold,
        }
    
    def reset_stats(self):
        """Сбрасывает статистику."""
        self.stats = SampleWeightingStats()
        self._loss_history = []


__all__ = [
    "SampleWeighter",
    "SampleWeightingStats",
]
