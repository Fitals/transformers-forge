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
Conservative GCA — Консервативная экстраполяция градиентов
==========================================================

Gradient Coherence Acceleration с жёсткими параметрами безопасности.
Активируется только в очень стабильные фазы обучения.
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, Any, Dict

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class GCAStats:
    """Статистика GCA."""
    total_steps: int = 0
    virtual_steps: int = 0
    checkpoints: int = 0
    rollbacks: int = 0
    average_coherence: float = 0.0


class ConservativeGCA:
    """
    Conservative Gradient Coherence Acceleration.
    
    Позволяет делать виртуальные оптимизационные шаги
    когда градиенты последовательных батчей согласованы.
    
    Консервативные параметры:
    - coherence_threshold = 0.95 (очень высокий)
    - max_virtual_steps = 2 (не более 2)
    - checkpoint_every = 5 (частые проверки)
    
    Args:
        coherence_threshold: Порог cos similarity для активации
        max_virtual_steps: Максимум виртуальных шагов подряд
        checkpoint_every: Частота полной проверки
        loss_tolerance: Допустимое отклонение loss
    """
    
    def __init__(
        self,
        coherence_threshold: float = 0.95,
        max_virtual_steps: int = 2,
        checkpoint_every: int = 5,
        loss_tolerance: float = 0.05,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("GCA требует PyTorch")
        
        self.coherence_threshold = coherence_threshold
        self.max_virtual_steps = max_virtual_steps
        self.checkpoint_every = checkpoint_every
        self.loss_tolerance = loss_tolerance
        
        # Состояние
        self._prev_gradient: Optional[Any] = None
        self._current_coherence: float = 0.0
        self._virtual_step_count: int = 0
        self._steps_since_checkpoint: int = 0
        self._expected_loss: float = 0.0
        self._last_real_loss: float = 0.0
        
        # История для анализа
        self._coherence_history: deque = deque(maxlen=20)
        self._loss_history: deque = deque(maxlen=20)
        
        # Статистика
        self.stats = GCAStats()
        
        # Флаг активности
        self._is_active = False
    
    def compute_coherence(
        self,
        current_gradient: Any,  # Flattened gradient tensor
    ) -> float:
        """
        Вычисляет когерентность с предыдущим градиентом.
        
        Args:
            current_gradient: Текущий градиент (flattened)
            
        Returns:
            Косинусное сходство [-1, 1]
        """
        if self._prev_gradient is None:
            return 0.0
        
        # Cosine similarity
        dot = torch.dot(current_gradient.flatten(), self._prev_gradient.flatten())
        norm_curr = current_gradient.norm()
        norm_prev = self._prev_gradient.norm()
        
        if norm_curr < 1e-8 or norm_prev < 1e-8:
            return 0.0
        
        coherence = (dot / (norm_curr * norm_prev)).item()
        
        return coherence
    
    def update(
        self,
        gradient: Any,
        loss: float,
    ) -> Tuple[bool, int]:
        """
        Обновляет состояние GCA и решает о виртуальных шагах.
        
        Args:
            gradient: Текущий градиент (flattened или dict)
            loss: Текущий loss
            
        Returns:
            (can_virtual_step, num_steps): 
                can_virtual_step - можно ли сделать виртуальный шаг
                num_steps - сколько шагов рекомендуется (1 или 2)
        """
        self.stats.total_steps += 1
        self._steps_since_checkpoint += 1
        
        # Flatten gradient if needed
        if isinstance(gradient, dict):
            flat_grad = torch.cat([g.flatten() for g in gradient.values()])
        else:
            flat_grad = gradient.flatten()
        
        # Вычисляем когерентность
        coherence = self.compute_coherence(flat_grad)
        self._current_coherence = coherence
        self._coherence_history.append(coherence)
        self._loss_history.append(loss)
        
        # Обновляем среднюю когерентность
        alpha = 0.1
        self.stats.average_coherence = (
            alpha * coherence + 
            (1 - alpha) * self.stats.average_coherence
        )
        
        # Сохраняем для следующего шага
        self._prev_gradient = flat_grad.clone()
        self._last_real_loss = loss
        
        # Проверяем условия для виртуального шага
        can_virtual = self._check_virtual_step_conditions(coherence, loss)
        
        if can_virtual:
            num_steps = self._calculate_virtual_steps(coherence)
            return True, num_steps
        else:
            return False, 0
    
    def _check_virtual_step_conditions(
        self,
        coherence: float,
        loss: float,
    ) -> bool:
        """Проверяет все условия для виртуального шага."""
        
        # 1. Достаточная когерентность
        if coherence < self.coherence_threshold:
            self._virtual_step_count = 0
            return False
        
        # 2. Не слишком много виртуальных шагов подряд
        if self._virtual_step_count >= self.max_virtual_steps:
            self._virtual_step_count = 0
            return False
        
        # 3. Время для checkpoint?
        if self._steps_since_checkpoint >= self.checkpoint_every:
            self._do_checkpoint(loss)
            return False
        
        # 4. Loss не вырос слишком сильно
        if self._expected_loss > 0:
            loss_ratio = loss / (self._expected_loss + 1e-8)
            if loss_ratio > 1.0 + self.loss_tolerance:
                self._do_rollback()
                return False
        
        return True
    
    def _calculate_virtual_steps(self, coherence: float) -> int:
        """Вычисляет количество виртуальных шагов."""
        
        # Чем выше когерентность, тем больше шагов (но max 2)
        if coherence > 0.98:
            return min(2, self.max_virtual_steps)
        elif coherence > 0.96:
            return min(1, self.max_virtual_steps)
        else:
            return 1
    
    def _do_checkpoint(self, loss: float):
        """Выполняет checkpoint — сброс счётчиков."""
        self.stats.checkpoints += 1
        self._steps_since_checkpoint = 0
        self._virtual_step_count = 0
        self._expected_loss = loss
    
    def _do_rollback(self):
        """Выполняет rollback — отключение виртуальных шагов."""
        self.stats.rollbacks += 1
        self._virtual_step_count = 0
        self._expected_loss = 0.0
    
    def record_virtual_step(self):
        """Записывает виртуальный шаг."""
        self._virtual_step_count += 1
        self.stats.virtual_steps += 1
    
    def get_expected_gradient_direction(self) -> Optional[Any]:
        """
        Возвращает ожидаемое направление градиента для виртуального шага.
        
        Returns:
            Нормализованный градиент или None
        """
        if self._prev_gradient is None:
            return None
        
        # Возвращаем нормализованное направление
        norm = self._prev_gradient.norm()
        if norm < 1e-8:
            return None
        
        return self._prev_gradient / norm
    
    def get_stats(self) -> dict:
        """Возвращает статистику GCA."""
        total = max(1, self.stats.total_steps)
        
        return {
            "total_steps": self.stats.total_steps,
            "virtual_steps": self.stats.virtual_steps,
            "virtual_step_rate": self.stats.virtual_steps / total,
            "checkpoints": self.stats.checkpoints,
            "rollbacks": self.stats.rollbacks,
            "average_coherence": self.stats.average_coherence,
            "current_coherence": self._current_coherence,
        }
    
    def reset(self):
        """Сбрасывает состояние GCA."""
        self._prev_gradient = None
        self._current_coherence = 0.0
        self._virtual_step_count = 0
        self._steps_since_checkpoint = 0
        self._expected_loss = 0.0
        self._coherence_history.clear()
        self._loss_history.clear()
        self.stats = GCAStats()


__all__ = [
    "ConservativeGCA",
    "GCAStats",
]
