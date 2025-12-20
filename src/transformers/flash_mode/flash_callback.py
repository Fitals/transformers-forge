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
Flash Mode Callback — Интеграция с Trainer
==========================================

Trainer callback для автоматической активации Flash Mode.
"""

from typing import Optional, Dict, Any
from datetime import datetime

try:
    from transformers import TrainerCallback, TrainerState, TrainerControl
    from transformers.training_args import TrainingArguments
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    TrainerCallback = object

from .alas import ALASTracker
from .sample_weighter import SampleWeighter
from .gca import ConservativeGCA
from .stats import FlashModeStats


class FlashModeCallback(TrainerCallback):
    """
    Trainer Callback для Flash Mode.
    
    Автоматически активирует и управляет Flash Mode во время обучения.
    
    Args:
        config: FlashConfig с параметрами
        verbose: Выводить статистику в консоль
    """
    
    def __init__(
        self,
        config: Optional[Any] = None,  # FlashConfig
        verbose: bool = True,
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("FlashModeCallback требует transformers")
        
        # Импортируем config здесь чтобы избежать circular import
        if config is None:
            from . import FlashConfig
            config = FlashConfig()
        
        self.config = config
        self.verbose = verbose
        
        # Компоненты
        self.alas_tracker: Optional[ALASTracker] = None
        self.sample_weighter: Optional[SampleWeighter] = None
        self.gca: Optional[ConservativeGCA] = None
        
        # Статистика
        self.stats = FlashModeStats()
        
        # Состояние
        self._is_active = False
        self._warmup_complete = False
        self._current_step = 0
        self._last_loss = float("inf")
        self._loss_baseline = None
        
        # Инициализация компонентов
        self._init_components()
    
    def _init_components(self):
        """Инициализирует компоненты Flash Mode."""
        
        if self.config.enable_alas:
            self.alas_tracker = ALASTracker(
                min_activity=self.config.min_activity,
                honest_check_every=self.config.honest_check_every,
                activity_decay=self.config.activity_decay,
            )
        
        if self.config.enable_sample_weighting:
            self.sample_weighter = SampleWeighter(
                min_weight=self.config.min_sample_weight,
                low_loss_threshold=self.config.low_loss_threshold,
            )
        
        if self.config.enable_gca:
            self.gca = ConservativeGCA(
                coherence_threshold=self.config.coherence_threshold,
                max_virtual_steps=self.config.max_virtual_steps,
                checkpoint_every=self.config.gca_checkpoint_every,
            )
    
    def on_train_begin(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        model=None,
        **kwargs,
    ):
        """Вызывается в начале обучения."""
        
        self.stats.start_time = datetime.now()
        self.stats.total_steps = 0
        self._current_step = 0
        
        # Инициализируем ALAS для модели
        if self.alas_tracker and model:
            self.alas_tracker.initialize(model)
        
        if self.verbose:
            print()
            print("╔══════════════════════════════════════════════════════════════════════╗")
            print("║  ⚡ FLASH MODE ACTIVATED                                             ║")
            print("╠══════════════════════════════════════════════════════════════════════╣")
            print(f"║  ALAS: {'ON' if self.config.enable_alas else 'OFF'}                                                           ║")
            print(f"║  Sample Weighting: {'ON' if self.config.enable_sample_weighting else 'OFF'}                                            ║")
            print(f"║  GCA: {'ON' if self.config.enable_gca else 'OFF'}                                                            ║")
            print(f"║  Warmup Steps: {self.config.warmup_steps}                                                   ║")
            print("╚══════════════════════════════════════════════════════════════════════╝")
            print()
    
    def on_step_end(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        model=None,
        **kwargs,
    ):
        """Вызывается после каждого шага."""
        
        self._current_step = state.global_step
        self.stats.total_steps = state.global_step
        
        # Проверяем окончание warmup
        if not self._warmup_complete and state.global_step >= self.config.warmup_steps:
            self._warmup_complete = True
            self.stats.warmup_complete = True
            
            if self.verbose:
                print(f"   ⚡ Flash Mode: Warmup complete at step {state.global_step}")
        
        # Обновляем метрики ALAS
        if self.alas_tracker and model and self._warmup_complete:
            self.alas_tracker.update_metrics(model)
            self.alas_tracker.step()
        
        # Проверяем loss spike
        if state.log_history:
            current_loss = state.log_history[-1].get("loss", self._last_loss)
            
            if self._loss_baseline is None:
                self._loss_baseline = current_loss
            
            # Проверка spike
            if current_loss > self._loss_baseline * self.config.loss_spike_threshold:
                self.stats.loss_spikes += 1
                
                if self.config.auto_disable_on_spike:
                    self._handle_loss_spike()
            else:
                # Обновляем baseline с decay
                self._loss_baseline = 0.95 * self._loss_baseline + 0.05 * current_loss
            
            self._last_loss = current_loss
            self.stats.loss_history.append(current_loss)
        
        # Периодический лог
        if self.verbose and state.global_step % self.config.log_every == 0:
            self._log_progress()
    
    def on_train_end(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs,
    ):
        """Вызывается в конце обучения."""
        
        # Собираем финальную статистику
        if self.alas_tracker:
            self.stats.update_from_components(alas_stats=self.alas_tracker.get_stats())
        
        if self.sample_weighter:
            self.stats.update_from_components(sw_stats=self.sample_weighter.get_stats())
        
        if self.gca:
            self.stats.update_from_components(gca_stats=self.gca.get_stats())
        
        if self.verbose:
            print(self.stats.format_summary())
    
    def _handle_loss_spike(self):
        """Обрабатывает spike loss."""
        
        self.stats.auto_disables += 1
        
        if self.verbose:
            print(f"   ⚠️ Flash Mode: Loss spike detected at step {self._current_step}")
        
        # Временно повышаем activity всех слоёв
        if self.alas_tracker:
            for state in self.alas_tracker.layer_states.values():
                state.activity_level = 1.0
        
        # Сбрасываем GCA
        if self.gca:
            self.gca.reset()
    
    def _log_progress(self):
        """Логирует прогресс."""
        
        if self.alas_tracker:
            stats = self.alas_tracker.get_stats()
            interp_rate = stats.get("interpolation_rate", 0) * 100
            avg_activity = stats.get("average_activity", 1.0)
            
            print(
                f"   ⚡ Step {self._current_step}: "
                f"ALAS interp={interp_rate:.1f}%, "
                f"activity={avg_activity:.2f}"
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает текущую статистику."""
        
        # Обновляем из компонентов
        if self.alas_tracker:
            self.stats.update_from_components(alas_stats=self.alas_tracker.get_stats())
        
        if self.sample_weighter:
            self.stats.update_from_components(sw_stats=self.sample_weighter.get_stats())
        
        if self.gca:
            self.stats.update_from_components(gca_stats=self.gca.get_stats())
        
        return self.stats.to_dict()


__all__ = [
    "FlashModeCallback",
]
