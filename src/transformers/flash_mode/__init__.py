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
Flash Mode — Режим ускоренного обучения для Transformers Forge
==============================================================

Flash Mode реализует технологию Adaptive Layer Activity Spectrum (ALAS),
которая динамически определяет уровень активности каждого слоя модели
и применяет интерполяцию градиентов для оптимизации вычислений.

Компоненты:
    - ALAS: Adaptive Layer Activity Spectrum
    - Sample Weighting: Адаптивное взвешивание примеров
    - Conservative GCA: Консервативная экстраполяция градиентов

Использование:
    >>> from transformers.flash_mode import FlashTrainer, FlashConfig
    >>> 
    >>> config = FlashConfig(enable_alas=True)
    >>> trainer = FlashTrainer(model=model, flash_config=config, ...)
    >>> trainer.train()

Ожидаемое ускорение: 1.3-1.5x без потери качества.

Добавлено в v1.1.3.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from .alas import (
    ALASTracker,
    LayerActivityState,
    GradientInterpolator,
)
from .sample_weighter import SampleWeighter
from .gca import ConservativeGCA
from .flash_callback import FlashModeCallback
from .stats import FlashModeStats


@dataclass
class FlashConfig:
    """
    Конфигурация Flash Mode.
    
    Args:
        enable_alas: Включить ALAS (Adaptive Layer Activity Spectrum)
        enable_sample_weighting: Включить взвешивание примеров
        enable_gca: Включить Conservative GCA
        
        # ALAS параметры
        min_activity: Минимальный уровень активности слоя (0.3 = 30% backward)
        honest_check_every: Частота полного честного backward
        activity_decay: Скорость изменения activity level
        
        # Sample Weighting параметры
        min_sample_weight: Минимальный вес примера
        low_loss_threshold: Порог "выученного" примера
        
        # GCA параметры
        coherence_threshold: Порог когерентности для виртуальных шагов
        max_virtual_steps: Максимум виртуальных шагов
        gca_checkpoint_every: Частота checkpoint для GCA
        
        # Safety параметры
        warmup_steps: Шаги разогрева перед активацией Flash Mode
        loss_spike_threshold: Порог роста loss для авто-отключения
        auto_disable_on_spike: Автоматически отключать при spike
    """
    
    # Компоненты
    enable_alas: bool = True
    enable_sample_weighting: bool = True
    enable_gca: bool = True
    
    # ALAS параметры
    min_activity: float = 0.3
    honest_check_every: int = 20
    activity_decay: float = 0.95
    
    # Sample Weighting параметры
    min_sample_weight: float = 0.3
    low_loss_threshold: float = 0.5
    
    # GCA параметры
    coherence_threshold: float = 0.95
    max_virtual_steps: int = 2
    gca_checkpoint_every: int = 5
    
    # Safety параметры
    warmup_steps: int = 100
    loss_spike_threshold: float = 1.1
    auto_disable_on_spike: bool = True
    
    # Logging
    verbose: bool = True
    log_every: int = 50


__all__ = [
    "FlashConfig",
    "ALASTracker",
    "LayerActivityState", 
    "GradientInterpolator",
    "SampleWeighter",
    "ConservativeGCA",
    "FlashModeCallback",
    "FlashModeStats",
]
