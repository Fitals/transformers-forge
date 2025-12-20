# Copyright 2024 Transformers Forge Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...

"""
Тесты для Flash Mode — режима ускоренного обучения.

Структура тестов:
1. FlashConfig - конфигурация
2. ALAS - Adaptive Layer Activity Spectrum
3. SampleWeighter - взвешивание примеров
4. ConservativeGCA - экстраполяция градиентов
5. Integration - интеграция компонентов
"""

import pytest
from unittest.mock import MagicMock, patch


class TestFlashConfig:
    """Тесты для FlashConfig."""
    
    def test_default_config(self):
        """Проверка дефолтных значений."""
        from transformers.flash_mode import FlashConfig
        
        config = FlashConfig()
        
        # Компоненты включены по умолчанию
        assert config.enable_alas is True
        assert config.enable_sample_weighting is True
        assert config.enable_gca is True
        
        # Консервативные параметры
        assert config.min_activity == 0.3
        assert config.coherence_threshold == 0.95
        assert config.max_virtual_steps == 2
    
    def test_custom_config(self):
        """Проверка кастомных значений."""
        from transformers.flash_mode import FlashConfig
        
        config = FlashConfig(
            enable_gca=False,
            min_activity=0.5,
            warmup_steps=200,
        )
        
        assert config.enable_gca is False
        assert config.min_activity == 0.5
        assert config.warmup_steps == 200


class TestALAS:
    """Тесты для ALAS."""
    
    def test_layer_activity_state(self):
        """Проверка состояния слоя."""
        from transformers.flash_mode.alas import LayerActivityState
        
        state = LayerActivityState(name="layer1")
        
        assert state.name == "layer1"
        assert state.activity_level == 1.0
        assert len(state.gradient_history) == 0
    
    def test_gradient_interpolator(self):
        """Проверка интерполяции градиентов."""
        torch = pytest.importorskip("torch")
        from transformers.flash_mode.alas import GradientInterpolator
        from collections import deque
        
        interpolator = GradientInterpolator(
            decay=0.9,
            noise_scale=0.0,  # Без шума для детерминированного теста
            use_trend=False,
        )
        
        # История из 3 градиентов
        history = deque(maxlen=10)
        history.append(torch.tensor([1.0, 2.0, 3.0]))
        history.append(torch.tensor([2.0, 3.0, 4.0]))
        history.append(torch.tensor([3.0, 4.0, 5.0]))
        
        result = interpolator.interpolate(history)
        
        assert result is not None
        assert result.shape == torch.Size([3])
    
    def test_alas_tracker_initialization(self):
        """Проверка инициализации трекера."""
        torch = pytest.importorskip("torch")
        import torch.nn as nn
        from transformers.flash_mode.alas import ALASTracker
        
        tracker = ALASTracker(min_activity=0.3)
        
        # Простая модель
        model = nn.Linear(10, 5)
        
        tracker.initialize(model)
        
        # Должны быть отслежены weight и bias
        assert len(tracker.layer_states) == 2
    
    def test_should_compute_backward(self):
        """Проверка решения о backward."""
        torch = pytest.importorskip("torch")
        import torch.nn as nn
        from transformers.flash_mode.alas import ALASTracker
        
        tracker = ALASTracker(
            min_activity=0.3,
            honest_check_every=10,
        )
        
        model = nn.Linear(10, 5)
        tracker.initialize(model)
        
        # Первый шаг — всегда честный
        layer_name = list(tracker.layer_states.keys())[0]
        result = tracker.should_compute_backward(layer_name)
        
        # Должен вернуть True или False (стохастически)
        assert isinstance(result, bool)


class TestSampleWeighter:
    """Тесты для SampleWeighter."""
    
    def test_weight_computation(self):
        """Проверка вычисления весов."""
        torch = pytest.importorskip("torch")
        from transformers.flash_mode.sample_weighter import SampleWeighter
        
        weighter = SampleWeighter(
            min_weight=0.3,
            low_loss_threshold=0.5,
            high_loss_threshold=1.5,
        )
        
        # Разные loss
        losses = torch.tensor([0.1, 0.5, 1.0, 2.0])
        weights = weighter.compute_weights(losses)
        
        assert weights.shape == losses.shape
        
        # Минимальный вес >= min_weight
        assert weights.min().item() >= 0.3
        
        # Максимальный вес <= 1.0
        assert weights.max().item() <= 1.0
        
        # Низкий loss → низкий вес
        assert weights[0].item() < weights[3].item()
    
    def test_stats_collection(self):
        """Проверка сбора статистики."""
        torch = pytest.importorskip("torch")
        from transformers.flash_mode.sample_weighter import SampleWeighter
        
        weighter = SampleWeighter(enable_stats=True)
        
        losses = torch.tensor([0.1, 0.5, 1.0])
        weighter.compute_weights(losses)
        
        stats = weighter.get_stats()
        
        assert stats["total_samples"] == 3
        assert "average_weight" in stats


class TestConservativeGCA:
    """Тесты для ConservativeGCA."""
    
    def test_coherence_computation(self):
        """Проверка вычисления когерентности."""
        torch = pytest.importorskip("torch")
        from transformers.flash_mode.gca import ConservativeGCA
        
        gca = ConservativeGCA(coherence_threshold=0.95)
        
        # Первый gradient — нет предыдущего
        grad1 = torch.tensor([1.0, 0.0, 0.0])
        can_virtual, _ = gca.update(grad1, loss=1.0)
        
        # Первый шаг — не может быть виртуальным
        assert can_virtual is False
        
        # Второй gradient — идентичный
        grad2 = torch.tensor([1.0, 0.0, 0.0])
        gca.update(grad2, loss=0.9)
        
        # Coherence должен быть 1.0
        stats = gca.get_stats()
        assert stats["current_coherence"] == pytest.approx(1.0, abs=0.01)
    
    def test_virtual_step_decision(self):
        """Проверка решения о виртуальном шаге."""
        torch = pytest.importorskip("torch")
        from transformers.flash_mode.gca import ConservativeGCA
        
        gca = ConservativeGCA(
            coherence_threshold=0.9,  # Снижаем для теста
            max_virtual_steps=2,
        )
        
        # Серия схожих градиентов
        for i in range(10):
            grad = torch.tensor([1.0, 0.1 * i, 0.0])
            can_virtual, num_steps = gca.update(grad, loss=1.0 - i*0.05)
        
        # Статистика должна отражать работу
        stats = gca.get_stats()
        assert stats["total_steps"] == 10


class TestFlashModeStats:
    """Тесты для FlashModeStats."""
    
    def test_metrics_computation(self):
        """Проверка вычисления метрик."""
        from transformers.flash_mode.stats import FlashModeStats
        
        stats = FlashModeStats()
        stats.total_steps = 100
        stats.alas_honest_backward = 60
        stats.alas_interpolated = 40
        stats.gca_virtual_steps = 10
        
        metrics = stats.compute_metrics()
        
        assert "effective_speedup" in metrics
        assert metrics["effective_speedup"] >= 1.0
        assert "alas_savings_percent" in metrics
    
    def test_summary_format(self):
        """Проверка форматирования сводки."""
        from transformers.flash_mode.stats import FlashModeStats
        
        stats = FlashModeStats()
        stats.total_steps = 100
        
        summary = stats.format_summary()
        
        assert "FLASH MODE" in summary
        assert "Effective Speedup" in summary


class TestIntegration:
    """Интеграционные тесты."""
    
    def test_full_import(self):
        """Проверка полного импорта."""
        from transformers.flash_mode import (
            FlashConfig,
            ALASTracker,
            SampleWeighter,
            ConservativeGCA,
            FlashModeCallback,
            FlashModeStats,
        )
        
        # Все классы импортированы
        assert FlashConfig is not None
        assert ALASTracker is not None
        assert SampleWeighter is not None
        assert ConservativeGCA is not None
        assert FlashModeCallback is not None
        assert FlashModeStats is not None
    
    def test_callback_creation(self):
        """Проверка создания callback."""
        from transformers.flash_mode import FlashConfig, FlashModeCallback
        
        config = FlashConfig(
            enable_alas=True,
            enable_sample_weighting=True,
            enable_gca=True,
        )
        
        callback = FlashModeCallback(config=config, verbose=False)
        
        assert callback.alas_tracker is not None
        assert callback.sample_weighter is not None
        assert callback.gca is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
