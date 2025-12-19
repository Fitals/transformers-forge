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
Learning Rate Finder — автоматический подбор оптимального learning rate.

Основан на работе Leslie Smith (2015):
"Cyclical Learning Rates for Training Neural Networks"

Использование:
    >>> from transformers.lr_finder import LRFinder
    >>> finder = LRFinder(model, train_dataloader)
    >>> optimal_lr = finder.find()
    >>> print(f"Рекомендуемый LR: {optimal_lr}")

    # Опционально: построить график
    >>> finder.plot("lr_finder.png")
"""

import copy
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .utils import logging


logger = logging.get_logger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class LRFinderResult:
    """Результат поиска оптимального learning rate.
    
    Attributes:
        optimal_lr: Рекомендуемый learning rate.
        min_lr: Минимальный тестируемый LR.
        max_lr: Максимальный тестируемый LR.
        num_steps: Количество шагов теста.
        lrs: Список всех протестированных LR.
        losses: Список loss на каждом шаге.
        smoothed_losses: Сглаженные значения loss.
        best_lr_idx: Индекс точки с минимальным loss.
        suggestion_method: Метод выбора ("steepest_gradient" или "minimum").
    """
    optimal_lr: float
    min_lr: float
    max_lr: float
    num_steps: int
    lrs: List[float] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    smoothed_losses: List[float] = field(default_factory=list)
    best_lr_idx: int = 0
    suggestion_method: str = "steepest_gradient"
    
    def __repr__(self) -> str:
        return (
            f"LRFinderResult(\n"
            f"  optimal_lr={self.optimal_lr:.2e},\n"
            f"  range=[{self.min_lr:.2e}, {self.max_lr:.2e}],\n"
            f"  num_steps={self.num_steps},\n"
            f"  method='{self.suggestion_method}'\n"
            f")"
        )


# =============================================================================
# LR Finder
# =============================================================================


class LRFinder:
    """
    Learning Rate Finder — автоматический подбор оптимального learning rate.
    
    Алгоритм (Leslie Smith, 2015):
    1. Начинает с очень маленького LR
    2. Экспоненциально увеличивает LR на каждом шаге
    3. Записывает loss на каждом шаге
    4. Находит LR где loss минимален или растёт быстрее всего
    5. Рекомендует LR немного ниже этой точки
    
    Args:
        model: PyTorch модель для тестирования.
        train_dataloader: DataLoader с обучающими данными.
        optimizer: Опциональный оптимизатор (по умолчанию AdamW).
        criterion: Опциональная функция loss.
        device: Устройство для обучения ("auto", "cuda", "cpu").
        
    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> from transformers.lr_finder import LRFinder
        >>> 
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> finder = LRFinder(model, train_dataloader)
        >>> result = finder.find()
        >>> print(f"Оптимальный LR: {result.optimal_lr}")
        >>> finder.plot("lr_curve.png")
    """
    
    def __init__(
        self,
        model: "torch.nn.Module",
        train_dataloader: "torch.utils.data.DataLoader",
        optimizer: Optional["torch.optim.Optimizer"] = None,
        criterion: Optional[callable] = None,
        device: str = "auto",
    ):
        try:
            import torch
        except ImportError:
            raise ImportError(
                "LRFinder требует PyTorch. "
                "Установите его командой: pip install torch"
            )
        
        self.torch = torch
        self.model = model
        self.train_dataloader = train_dataloader
        self.criterion = criterion
        self.device = self._get_device(device)
        
        # Сохраняем оригинальное состояние модели
        self._original_state = copy.deepcopy(model.state_dict())
        
        # Результаты
        self.result: Optional[LRFinderResult] = None
        self._lrs: List[float] = []
        self._losses: List[float] = []
        
        # Создаём оптимизатор если не передан
        if optimizer is None:
            self._optimizer_class = torch.optim.AdamW
            self._optimizer_kwargs = {"weight_decay": 0.01}
        else:
            self._optimizer = optimizer
            self._optimizer_class = None
        
        logger.info(
            f"LRFinder initialized: "
            f"model={type(model).__name__}, "
            f"device={self.device}"
        )
    
    def _get_device(self, device: str) -> "torch.device":
        """Определяет устройство для обучения."""
        if device == "auto":
            if self.torch.cuda.is_available():
                return self.torch.device("cuda")
            elif hasattr(self.torch.backends, "mps") and self.torch.backends.mps.is_available():
                return self.torch.device("mps")
            else:
                return self.torch.device("cpu")
        return self.torch.device(device)
    
    def _create_optimizer(self, lr: float) -> "torch.optim.Optimizer":
        """Создаёт оптимизатор с заданным LR."""
        if self._optimizer_class is not None:
            return self._optimizer_class(
                self.model.parameters(),
                lr=lr,
                **self._optimizer_kwargs
            )
        else:
            # Обновляем LR существующего оптимизатора
            for param_group in self._optimizer.param_groups:
                param_group["lr"] = lr
            return self._optimizer
    
    def _compute_loss(
        self,
        batch: Dict[str, Any],
    ) -> "torch.Tensor":
        """Вычисляет loss для батча."""
        # Перемещаем батч на устройство
        batch = {
            k: v.to(self.device) if hasattr(v, "to") else v
            for k, v in batch.items()
        }
        
        # Forward pass
        outputs = self.model(**batch)
        
        # Получаем loss
        if self.criterion is not None:
            loss = self.criterion(outputs, batch)
        elif hasattr(outputs, "loss") and outputs.loss is not None:
            loss = outputs.loss
        else:
            raise ValueError(
                "Модель не возвращает loss. "
                "Передайте criterion в LRFinder или используйте модель с встроенным loss."
            )
        
        return loss
    
    def _smooth_losses(
        self,
        losses: List[float],
        beta: float = 0.98,
    ) -> List[float]:
        """Сглаживает loss с помощью exponential moving average."""
        smoothed = []
        avg_loss = 0.0
        
        for i, loss in enumerate(losses):
            avg_loss = beta * avg_loss + (1 - beta) * loss
            # Коррекция смещения (bias correction)
            smoothed.append(avg_loss / (1 - beta ** (i + 1)))
        
        return smoothed
    
    def _find_steep_gradient(
        self,
        lrs: List[float],
        smoothed_losses: List[float],
    ) -> int:
        """Находит точку с максимальным отрицательным градиентом loss."""
        if len(lrs) < 3:
            return 0
        
        gradients = []
        for i in range(1, len(smoothed_losses)):
            # Градиент в логарифмическом пространстве LR
            grad = (smoothed_losses[i] - smoothed_losses[i - 1]) / (
                math.log10(lrs[i]) - math.log10(lrs[i - 1])
            )
            gradients.append(grad)
        
        # Находим точку с минимальным (самым отрицательным) градиентом
        min_grad_idx = gradients.index(min(gradients))
        
        return min_grad_idx
    
    def find(
        self,
        min_lr: float = 1e-8,
        max_lr: float = 1e-1,
        num_steps: int = 100,
        smooth_factor: float = 0.98,
        divergence_threshold: float = 4.0,
        suggestion_method: str = "steepest_gradient",
    ) -> LRFinderResult:
        """
        Запускает поиск оптимального learning rate.
        
        Args:
            min_lr: Минимальный LR для тестирования.
            max_lr: Максимальный LR для тестирования.
            num_steps: Количество шагов теста.
            smooth_factor: Коэффициент сглаживания loss (0-1).
            divergence_threshold: Порог для остановки при взрыве loss.
            suggestion_method: Метод выбора LR:
                - "steepest_gradient": LR где loss падает быстрее всего
                - "minimum": LR с минимальным loss
                
        Returns:
            LRFinderResult с рекомендуемым LR и историей.
            
        Example:
            >>> result = finder.find(min_lr=1e-7, max_lr=1e-2, num_steps=50)
            >>> print(f"Оптимальный LR: {result.optimal_lr:.2e}")
        """
        logger.info(
            f"Starting LR search: "
            f"range=[{min_lr:.2e}, {max_lr:.2e}], "
            f"steps={num_steps}"
        )
        
        # Восстанавливаем оригинальные веса
        self.model.load_state_dict(copy.deepcopy(self._original_state))
        self.model.to(self.device)
        self.model.train()
        
        # Вычисляем множитель для экспоненциального роста LR
        lr_mult = (max_lr / min_lr) ** (1 / num_steps)
        
        # Инициализация
        current_lr = min_lr
        optimizer = self._create_optimizer(current_lr)
        
        self._lrs = []
        self._losses = []
        best_loss = float("inf")
        
        # Создаём итератор по данным
        data_iter = iter(self.train_dataloader)
        
        print(f"\n🔍 LR Finder: поиск оптимального learning rate...")
        print(f"   Диапазон: [{min_lr:.2e}, {max_lr:.2e}]")
        print(f"   Шаги: {num_steps}")
        print()
        
        for step in range(num_steps):
            # Получаем батч (с перезапуском итератора если нужно)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)
            
            # Forward + backward
            optimizer.zero_grad()
            
            try:
                loss = self._compute_loss(batch)
                loss.backward()
                optimizer.step()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    warnings.warn(
                        f"OOM на шаге {step} с LR={current_lr:.2e}. "
                        "Попробуйте уменьшить batch size."
                    )
                    break
                raise
            
            # Записываем результат
            loss_value = loss.item()
            self._lrs.append(current_lr)
            self._losses.append(loss_value)
            
            # Обновляем лучший loss
            if loss_value < best_loss:
                best_loss = loss_value
            
            # Проверяем на взрыв loss
            if loss_value > best_loss * divergence_threshold:
                logger.info(
                    f"Loss diverged at step {step} "
                    f"(loss={loss_value:.4f} > {divergence_threshold}x best)"
                )
                print(f"   ⚠️ Loss взорвался на шаге {step}, останавливаем поиск")
                break
            
            # Прогресс
            if (step + 1) % 10 == 0 or step == 0:
                print(f"   Шаг {step + 1}/{num_steps}: LR={current_lr:.2e}, Loss={loss_value:.4f}")
            
            # Увеличиваем LR
            current_lr *= lr_mult
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr
        
        # Сглаживаем losses
        smoothed = self._smooth_losses(self._losses, smooth_factor)
        
        # Находим оптимальный LR
        if suggestion_method == "steepest_gradient":
            best_idx = self._find_steep_gradient(self._lrs, smoothed)
            # Берём LR немного левее точки максимального спуска
            optimal_idx = max(0, best_idx - 1)
        else:  # minimum
            optimal_idx = smoothed.index(min(smoothed))
        
        optimal_lr = self._lrs[optimal_idx]
        
        # Применяем коэффициент безопасности (берём LR на порядок меньше)
        suggested_lr = optimal_lr / 10
        
        # Создаём результат
        self.result = LRFinderResult(
            optimal_lr=suggested_lr,
            min_lr=min_lr,
            max_lr=max_lr,
            num_steps=len(self._lrs),
            lrs=self._lrs.copy(),
            losses=self._losses.copy(),
            smoothed_losses=smoothed,
            best_lr_idx=optimal_idx,
            suggestion_method=suggestion_method,
        )
        
        # Восстанавливаем оригинальные веса
        self.model.load_state_dict(copy.deepcopy(self._original_state))
        
        print()
        print(f"   ✅ Рекомендуемый LR: {suggested_lr:.2e}")
        print(f"      (на основе анализа {len(self._lrs)} шагов)")
        print()
        
        logger.info(f"LR search complete: optimal_lr={suggested_lr:.2e}")
        
        return self.result
    
    def plot(
        self,
        output_path: Optional[str] = None,
        log_scale: bool = True,
        show_suggestion: bool = True,
        skip_start: int = 5,
        skip_end: int = 5,
    ) -> Optional[str]:
        """
        Строит график зависимости loss от learning rate.
        
        Args:
            output_path: Путь для сохранения графика (None = показать).
            log_scale: Использовать логарифмическую шкалу для LR.
            show_suggestion: Показать вертикальную линию для рекомендуемого LR.
            skip_start: Пропустить первые N точек (обычно шумные).
            skip_end: Пропустить последние N точек (обычно взрывные).
            
        Returns:
            Путь к сохранённому графику или None.
            
        Example:
            >>> finder.plot("lr_curve.png")
        """
        if self.result is None:
            raise ValueError(
                "Сначала запустите find() для поиска LR."
            )
        
        try:
            import matplotlib
            matplotlib.use("Agg")  # Для работы без GUI
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn(
                "matplotlib не установлен. "
                "Установите его командой: pip install matplotlib"
            )
            return None
        
        # Подготавливаем данные
        lrs = self.result.lrs[skip_start:-skip_end] if skip_end > 0 else self.result.lrs[skip_start:]
        losses = self.result.smoothed_losses[skip_start:-skip_end] if skip_end > 0 else self.result.smoothed_losses[skip_start:]
        
        # Создаём график
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(lrs, losses, linewidth=2, color="#2E86AB", label="Loss (smoothed)")
        
        if show_suggestion and self.result.optimal_lr:
            ax.axvline(
                x=self.result.optimal_lr,
                color="#E94F37",
                linestyle="--",
                linewidth=2,
                label=f"Рекомендуемый LR: {self.result.optimal_lr:.2e}"
            )
        
        if log_scale:
            ax.set_xscale("log")
        
        ax.set_xlabel("Learning Rate", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("LR Finder — Transformers Forge", fontsize=14, fontweight="bold")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        
        # Стилизация
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        plt.tight_layout()
        
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"LR plot saved to {path}")
            return str(path)
        else:
            plt.show()
            return None
    
    def reset(self):
        """Сбрасывает результаты и восстанавливает оригинальные веса модели."""
        self.model.load_state_dict(copy.deepcopy(self._original_state))
        self.result = None
        self._lrs = []
        self._losses = []
        logger.info("LRFinder reset")
    
    def get_suggestion(self) -> float:
        """Возвращает рекомендуемый LR без перезапуска поиска."""
        if self.result is None:
            raise ValueError(
                "Сначала запустите find() для поиска LR."
            )
        return self.result.optimal_lr


# =============================================================================
# Convenience functions
# =============================================================================


def find_optimal_lr(
    model: "torch.nn.Module",
    train_dataloader: "torch.utils.data.DataLoader",
    min_lr: float = 1e-8,
    max_lr: float = 1e-1,
    num_steps: int = 100,
    device: str = "auto",
    plot_path: Optional[str] = None,
) -> float:
    """
    Быстрый способ найти оптимальный learning rate.
    
    Args:
        model: PyTorch модель.
        train_dataloader: DataLoader с обучающими данными.
        min_lr: Минимальный LR для тестирования.
        max_lr: Максимальный LR для тестирования.
        num_steps: Количество шагов теста.
        device: Устройство ("auto", "cuda", "cpu").
        plot_path: Путь для сохранения графика (опционально).
        
    Returns:
        Рекомендуемый learning rate.
        
    Example:
        >>> optimal_lr = find_optimal_lr(model, train_dataloader)
        >>> print(f"Используйте LR: {optimal_lr:.2e}")
    """
    finder = LRFinder(model, train_dataloader, device=device)
    result = finder.find(min_lr=min_lr, max_lr=max_lr, num_steps=num_steps)
    
    if plot_path:
        finder.plot(plot_path)
    
    return result.optimal_lr
