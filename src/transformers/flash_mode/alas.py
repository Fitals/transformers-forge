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
ALAS — Adaptive Layer Activity Spectrum
=======================================

Ядро Flash Mode. Определяет уровень активности каждого слоя модели
и решает: честный backward или интерполяция градиента.

Компоненты:
    - LayerActivityState: Состояние отдельного слоя
    - ALASTracker: Отслеживание активности всех слоёв
    - GradientInterpolator: Интерполяция градиентов с трендом
"""

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class LayerActivityState:
    """
    Состояние активности отдельного слоя.
    
    Attributes:
        name: Имя слоя
        activity_level: Текущий уровень активности (0.3 - 1.0)
        gradient_history: История последних градиентов
        magnitude_history: История magnitude градиентов
        last_honest_step: Последний шаг с честным backward
        interpolation_count: Количество интерполяций подряд
    """
    name: str
    activity_level: float = 1.0
    gradient_history: deque = field(default_factory=lambda: deque(maxlen=10))
    magnitude_history: deque = field(default_factory=lambda: deque(maxlen=20))
    variance_history: deque = field(default_factory=lambda: deque(maxlen=10))
    last_honest_step: int = 0
    interpolation_count: int = 0
    
    # Метрики
    heat: float = 1.0
    stability: float = 0.0
    
    def __post_init__(self):
        if not isinstance(self.gradient_history, deque):
            self.gradient_history = deque(maxlen=10)
        if not isinstance(self.magnitude_history, deque):
            self.magnitude_history = deque(maxlen=20)
        if not isinstance(self.variance_history, deque):
            self.variance_history = deque(maxlen=10)


class GradientInterpolator:
    """
    Интерполятор градиентов с учётом тренда.
    
    Вместо простого использования последнего градиента,
    вычисляет взвешенное среднее с экспоненциальным затуханием
    и добавляет небольшой шум для exploration.
    """
    
    def __init__(
        self,
        decay: float = 0.9,
        noise_scale: float = 0.05,
        use_trend: bool = True,
    ):
        self.decay = decay
        self.noise_scale = noise_scale
        self.use_trend = use_trend
    
    def interpolate(
        self,
        gradient_history: deque,
        device: Optional[Any] = None,
    ) -> Optional[Any]:
        """
        Интерполирует градиент на основе истории.
        
        Args:
            gradient_history: История градиентов (deque of tensors)
            device: Устройство для результата
            
        Returns:
            Интерполированный градиент или None если история пуста
        """
        if not TORCH_AVAILABLE:
            return None
            
        if len(gradient_history) == 0:
            return None
        
        if len(gradient_history) == 1:
            return gradient_history[0].clone()
        
        # Вычисляем веса с экспоненциальным затуханием
        n = len(gradient_history)
        weights = []
        for i in range(n):
            w = self.decay ** (n - 1 - i)
            weights.append(w)
        
        # Нормализуем веса
        total = sum(weights)
        weights = [w / total for w in weights]
        
        # Взвешенное среднее
        result = torch.zeros_like(gradient_history[0])
        for i, grad in enumerate(gradient_history):
            result += weights[i] * grad
        
        # Добавляем тренд если включено
        if self.use_trend and n >= 2:
            trend = gradient_history[-1] - gradient_history[-2]
            result += 0.2 * trend  # Небольшой вклад тренда
        
        # Добавляем шум для exploration
        if self.noise_scale > 0:
            noise = torch.randn_like(result) * result.abs().mean() * self.noise_scale
            result += noise
        
        return result


class ALASTracker:
    """
    Adaptive Layer Activity Spectrum Tracker.
    
    Отслеживает активность всех слоёв модели и принимает решения
    о том, выполнять честный backward или использовать интерполяцию.
    
    Args:
        min_activity: Минимальный уровень активности (floor)
        honest_check_every: Частота принудительного честного backward
        activity_decay: Скорость изменения activity level
        heat_threshold: Порог heat для определения активности
    """
    
    def __init__(
        self,
        min_activity: float = 0.3,
        honest_check_every: int = 20,
        activity_decay: float = 0.95,
        heat_threshold: float = 0.1,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("ALAS требует PyTorch")
        
        self.min_activity = min_activity
        self.honest_check_every = honest_check_every
        self.activity_decay = activity_decay
        self.heat_threshold = heat_threshold
        
        # Состояние слоёв
        self.layer_states: Dict[str, LayerActivityState] = {}
        
        # Интерполятор
        self.interpolator = GradientInterpolator()
        
        # Счётчик шагов
        self.global_step = 0
        
        # Статистика
        self.total_honest = 0
        self.total_interpolated = 0
        
        # Флаг инициализации
        self._initialized = False
    
    def initialize(self, model: nn.Module):
        """
        Инициализирует tracker для модели.
        
        Args:
            model: PyTorch модель
        """
        self.layer_states.clear()
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.layer_states[name] = LayerActivityState(name=name)
        
        self._initialized = True
    
    def update_metrics(self, model: nn.Module):
        """
        Обновляет метрики для всех слоёв после backward.
        
        Args:
            model: Модель после backward pass
        """
        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            
            if name not in self.layer_states:
                self.layer_states[name] = LayerActivityState(name=name)
            
            state = self.layer_states[name]
            
            # Вычисляем magnitude
            grad_magnitude = param.grad.abs().mean().item()
            weight_magnitude = param.abs().mean().item() + 1e-8
            
            # Heat = насколько активен слой
            heat = grad_magnitude / weight_magnitude
            state.heat = self.activity_decay * state.heat + (1 - self.activity_decay) * heat
            
            # Сохраняем в историю
            state.magnitude_history.append(grad_magnitude)
            
            # Вычисляем variance (стабильность)
            if len(state.magnitude_history) >= 3:
                recent = list(state.magnitude_history)[-5:]
                variance = sum((x - sum(recent)/len(recent))**2 for x in recent) / len(recent)
                state.variance_history.append(variance)
                
                # Stability = 1 / (1 + variance)
                avg_var = sum(state.variance_history) / len(state.variance_history)
                state.stability = 1.0 / (1.0 + avg_var * 100)
            
            # Обновляем activity_level
            # Высокий heat + низкая stability = высокая активность
            # Низкий heat + высокая stability = низкая активность
            raw_activity = state.heat / (self.heat_threshold + state.heat)
            raw_activity *= (1.0 + (1.0 - state.stability) * 0.5)
            
            # Применяем decay к текущему activity
            new_activity = self.activity_decay * state.activity_level + (1 - self.activity_decay) * raw_activity
            
            # Clamp к [min_activity, 1.0]
            state.activity_level = max(self.min_activity, min(1.0, new_activity))
    
    def should_compute_backward(self, layer_name: str) -> bool:
        """
        Решает: выполнять честный backward для слоя или интерполировать.
        
        Args:
            layer_name: Имя слоя
            
        Returns:
            True = честный backward, False = интерполяция
        """
        if layer_name not in self.layer_states:
            return True  # Новый слой — всегда честно
        
        state = self.layer_states[layer_name]
        
        # Принудительный честный backward каждые N шагов
        if self.global_step - state.last_honest_step >= self.honest_check_every:
            return True
        
        # Если слишком много интерполяций подряд — честный
        if state.interpolation_count >= 5:
            return True
        
        # Стохастическое решение на основе activity_level
        return random.random() < state.activity_level
    
    def record_decision(self, layer_name: str, was_honest: bool):
        """
        Записывает решение для статистики.
        
        Args:
            layer_name: Имя слоя
            was_honest: True если был честный backward
        """
        if layer_name not in self.layer_states:
            return
        
        state = self.layer_states[layer_name]
        
        if was_honest:
            state.last_honest_step = self.global_step
            state.interpolation_count = 0
            self.total_honest += 1
        else:
            state.interpolation_count += 1
            self.total_interpolated += 1
    
    def save_gradient(self, layer_name: str, gradient: Any):
        """
        Сохраняет градиент в историю для интерполяции.
        
        Args:
            layer_name: Имя слоя
            gradient: Тензор градиента
        """
        if layer_name not in self.layer_states:
            return
        
        # Сохраняем копию
        self.layer_states[layer_name].gradient_history.append(gradient.detach().clone())
    
    def get_interpolated_gradient(self, layer_name: str) -> Optional[Any]:
        """
        Получает интерполированный градиент для слоя.
        
        Args:
            layer_name: Имя слоя
            
        Returns:
            Интерполированный градиент или None
        """
        if layer_name not in self.layer_states:
            return None
        
        state = self.layer_states[layer_name]
        
        if len(state.gradient_history) == 0:
            return None
        
        return self.interpolator.interpolate(state.gradient_history)
    
    def step(self):
        """Увеличивает глобальный счётчик шагов."""
        self.global_step += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику ALAS.
        
        Returns:
            Словарь со статистикой
        """
        total = self.total_honest + self.total_interpolated
        
        return {
            "total_honest": self.total_honest,
            "total_interpolated": self.total_interpolated,
            "interpolation_rate": self.total_interpolated / max(1, total),
            "average_activity": sum(s.activity_level for s in self.layer_states.values()) / max(1, len(self.layer_states)),
            "layers_tracked": len(self.layer_states),
        }
    
    def get_layer_activities(self) -> Dict[str, float]:
        """
        Возвращает текущие activity levels всех слоёв.
        
        Returns:
            Словарь {layer_name: activity_level}
        """
        return {name: state.activity_level for name, state in self.layer_states.items()}


__all__ = [
    "LayerActivityState",
    "GradientInterpolator",
    "ALASTracker",
]
