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
Flash Mode Stats — Статистика и визуализация
=============================================

Сбор и отображение статистики Flash Mode.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime


@dataclass
class FlashModeStats:
    """
    Агрегированная статистика Flash Mode.
    
    Собирает данные со всех компонентов:
    - ALAS
    - Sample Weighting
    - Conservative GCA
    """
    
    # Общее
    start_time: Optional[datetime] = None
    total_steps: int = 0
    flash_active_steps: int = 0
    warmup_complete: bool = False
    
    # ALAS статистика
    alas_honest_backward: int = 0
    alas_interpolated: int = 0
    alas_average_activity: float = 1.0
    
    # Sample Weighting статистика
    sw_total_samples: int = 0
    sw_low_loss_samples: int = 0
    sw_average_weight: float = 1.0
    
    # GCA статистика
    gca_virtual_steps: int = 0
    gca_checkpoints: int = 0
    gca_rollbacks: int = 0
    gca_average_coherence: float = 0.0
    
    # Safety
    loss_spikes: int = 0
    auto_disables: int = 0
    
    # История loss
    loss_history: List[float] = field(default_factory=list)
    
    def update_from_components(
        self,
        alas_stats: Optional[Dict] = None,
        sw_stats: Optional[Dict] = None,
        gca_stats: Optional[Dict] = None,
    ):
        """Обновляет статистику из компонентов."""
        
        if alas_stats:
            self.alas_honest_backward = alas_stats.get("total_honest", 0)
            self.alas_interpolated = alas_stats.get("total_interpolated", 0)
            self.alas_average_activity = alas_stats.get("average_activity", 1.0)
        
        if sw_stats:
            self.sw_total_samples = sw_stats.get("total_samples", 0)
            self.sw_low_loss_samples = int(
                sw_stats.get("low_loss_rate", 0) * self.sw_total_samples
            )
            self.sw_average_weight = sw_stats.get("average_weight", 1.0)
        
        if gca_stats:
            self.gca_virtual_steps = gca_stats.get("virtual_steps", 0)
            self.gca_checkpoints = gca_stats.get("checkpoints", 0)
            self.gca_rollbacks = gca_stats.get("rollbacks", 0)
            self.gca_average_coherence = gca_stats.get("average_coherence", 0.0)
    
    def compute_metrics(self) -> Dict[str, Any]:
        """Вычисляет ключевые метрики."""
        
        # ALAS savings
        total_alas = self.alas_honest_backward + self.alas_interpolated
        alas_savings = self.alas_interpolated / max(1, total_alas)
        
        # Sample weighting savings
        sw_savings = 1.0 - self.sw_average_weight
        
        # GCA savings (грубая оценка)
        gca_savings = self.gca_virtual_steps / max(1, self.total_steps) * 0.5
        
        # Общее ускорение
        # Backward = 55% времени
        # ALAS экономит часть backward
        # GCA экономит полные шаги
        backward_savings = alas_savings * 0.55  # 55% от backward
        total_savings = backward_savings * 0.6 + gca_savings * 0.3 + sw_savings * 0.1
        
        effective_speedup = 1.0 / (1.0 - min(0.5, total_savings))
        
        return {
            "effective_speedup": round(effective_speedup, 2),
            "alas_savings_percent": round(alas_savings * 100, 1),
            "sw_average_weight": round(self.sw_average_weight, 3),
            "gca_virtual_rate": round(self.gca_virtual_steps / max(1, self.total_steps) * 100, 1),
            "total_steps": self.total_steps,
            "loss_spikes": self.loss_spikes,
            "auto_disables": self.auto_disables,
        }
    
    def format_summary(self) -> str:
        """Форматирует красивую сводку."""
        
        metrics = self.compute_metrics()
        
        lines = [
            "",
            "╔══════════════════════════════════════════════════════════════════════╗",
            "║  ⚡ FLASH MODE — STATISTICS                                          ║",
            "╠══════════════════════════════════════════════════════════════════════╣",
            f"║  Effective Speedup: {metrics['effective_speedup']:.2f}x                                          ║",
            "║                                                                      ║",
            "║  Components:                                                         ║",
            f"║    • ALAS Savings: {metrics['alas_savings_percent']:.1f}%                                          ║",
            f"║    • GCA Virtual Steps: {metrics['gca_virtual_rate']:.1f}%                                      ║",
            f"║    • Sample Weight Avg: {metrics['sw_average_weight']:.3f}                                      ║",
            "║                                                                      ║",
            f"║  Total Steps: {metrics['total_steps']}                                                 ║",
            f"║  Loss Spikes: {metrics['loss_spikes']} | Auto-Disables: {metrics['auto_disables']}                              ║",
            "╚══════════════════════════════════════════════════════════════════════╝",
            "",
        ]
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует в словарь."""
        return {
            "total_steps": self.total_steps,
            "flash_active_steps": self.flash_active_steps,
            "alas": {
                "honest_backward": self.alas_honest_backward,
                "interpolated": self.alas_interpolated,
                "average_activity": self.alas_average_activity,
            },
            "sample_weighting": {
                "total_samples": self.sw_total_samples,
                "low_loss_samples": self.sw_low_loss_samples,
                "average_weight": self.sw_average_weight,
            },
            "gca": {
                "virtual_steps": self.gca_virtual_steps,
                "checkpoints": self.gca_checkpoints,
                "rollbacks": self.gca_rollbacks,
                "average_coherence": self.gca_average_coherence,
            },
            "safety": {
                "loss_spikes": self.loss_spikes,
                "auto_disables": self.auto_disables,
            },
            "metrics": self.compute_metrics(),
        }


__all__ = [
    "FlashModeStats",
]
