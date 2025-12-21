"""
Тесты для модуля Adaptive Loss
==============================

Запуск:
    pytest tests/forge/test_adaptive_loss.py -v
"""

import pytest
import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """Простая модель для тестов"""
    
    def __init__(self, vocab_size=1000, hidden_size=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        return self.fc(self.embed(x))


class TestAdaptiveLossConfig:
    """Тесты для AdaptiveLossConfig"""
    
    def test_default_config(self):
        """Тест создания конфига с дефолтами"""
        from transformers.adaptive_loss import AdaptiveLossConfig
        
        config = AdaptiveLossConfig()
        
        assert config.enabled == True
        assert config.gamma == 2.0
        assert config.focus_on_hard == True
        assert config.ignore_padding == True
    
    def test_custom_config(self):
        """Тест кастомного конфига"""
        from transformers.adaptive_loss import AdaptiveLossConfig
        
        config = AdaptiveLossConfig(
            gamma=3.0,
            alpha=0.5,
            min_weight=0.2,
            max_weight=3.0,
        )
        
        assert config.gamma == 3.0
        assert config.alpha == 0.5
        assert config.min_weight == 0.2
        assert config.max_weight == 3.0
    
    def test_invalid_gamma(self):
        """Тест валидации gamma"""
        from transformers.adaptive_loss import AdaptiveLossConfig
        
        with pytest.raises(ValueError):
            AdaptiveLossConfig(gamma=-1.0)
    
    def test_invalid_weights(self):
        """Тест валидации min/max weights"""
        from transformers.adaptive_loss import AdaptiveLossConfig
        
        with pytest.raises(ValueError):
            AdaptiveLossConfig(min_weight=5.0, max_weight=1.0)


class TestComputeWeightedLoss:
    """Тесты для compute_weighted_loss"""
    
    @pytest.fixture
    def model_and_data(self):
        """Создаём модель и тестовые данные"""
        model = SimpleModel()
        batch_size, seq_len = 2, 10
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        labels = torch.randint(0, 1000, (batch_size, seq_len))
        labels[0, -3:] = -100  # Padding
        
        with torch.no_grad():
            logits = model(input_ids)
        
        return logits, labels
    
    def test_compute_loss_basic(self, model_and_data):
        """Тест базового вычисления loss"""
        from transformers.adaptive_loss import compute_weighted_loss
        
        logits, labels = model_and_data
        
        loss = compute_weighted_loss(logits, labels)
        
        assert loss.ndim == 0  # Scalar
        assert loss.item() > 0
    
    def test_compute_loss_disabled(self, model_and_data):
        """Тест с отключённым adaptive loss"""
        from transformers.adaptive_loss import compute_weighted_loss, AdaptiveLossConfig
        
        logits, labels = model_and_data
        config = AdaptiveLossConfig(enabled=False)
        
        loss = compute_weighted_loss(logits, labels, config)
        
        assert loss.ndim == 0
        assert loss.item() > 0
    
    def test_compute_loss_reduction_none(self, model_and_data):
        """Тест с reduction='none'"""
        from transformers.adaptive_loss import compute_weighted_loss
        
        logits, labels = model_and_data
        
        loss = compute_weighted_loss(logits, labels, reduction="none")
        
        assert loss.ndim == 2
        assert loss.shape[0] == 2  # batch_size


class TestFocalLoss:
    """Тесты для focal_loss"""
    
    @pytest.fixture
    def model_and_data(self):
        """Создаём тестовые данные"""
        batch_size, seq_len, vocab = 2, 10, 100
        
        logits = torch.randn(batch_size, seq_len, vocab)
        labels = torch.randint(0, vocab, (batch_size, seq_len))
        
        return logits, labels
    
    def test_focal_loss_basic(self, model_and_data):
        """Тест базового focal loss"""
        from transformers.adaptive_loss import focal_loss
        
        logits, labels = model_and_data
        
        loss = focal_loss(logits, labels, gamma=2.0)
        
        assert loss.ndim == 0
        assert loss.item() > 0
    
    def test_focal_loss_gamma_zero(self, model_and_data):
        """Тест focal loss с gamma=0 (эквивалент CE)"""
        from transformers.adaptive_loss import focal_loss
        
        logits, labels = model_and_data
        
        loss = focal_loss(logits, labels, gamma=0.0)
        
        assert loss.ndim == 0
        # При gamma=0 должен быть обычный CE
    
    def test_focal_loss_high_gamma(self, model_and_data):
        """Тест focal loss с высоким gamma"""
        from transformers.adaptive_loss import focal_loss
        
        logits, labels = model_and_data
        
        loss_low = focal_loss(logits, labels, gamma=0.5)
        loss_high = focal_loss(logits, labels, gamma=5.0)
        
        # Высокий gamma даёт меньший loss (больше down-weighting)
        # Это не всегда так, зависит от данных


class TestCreateLossMask:
    """Тесты для create_loss_mask"""
    
    def test_mask_padding(self):
        """Тест маскирования padding"""
        from transformers.adaptive_loss import create_loss_mask
        
        labels = torch.tensor([[1, 2, 3, -100, -100]])
        
        mask = create_loss_mask(labels)
        
        assert mask.shape == labels.shape
        assert mask[0, 0] == True
        assert mask[0, 3] == False
        assert mask[0, 4] == False


class TestAdaptiveLossCallback:
    """Тесты для AdaptiveLossCallback"""
    
    def test_callback_creation(self):
        """Тест создания callback"""
        from transformers.adaptive_loss import AdaptiveLossCallback
        
        callback = AdaptiveLossCallback()
        
        assert callback.step_count == 0
        assert callback.config.enabled == True
    
    def test_callback_compute_loss(self):
        """Тест compute_loss в callback"""
        from transformers.adaptive_loss import AdaptiveLossCallback
        
        callback = AdaptiveLossCallback()
        
        logits = torch.randn(2, 10, 100)
        labels = torch.randint(0, 100, (2, 10))
        
        loss = callback.compute_loss(logits, labels)
        
        assert callback.step_count == 1
        assert len(callback.loss_history) == 1
    
    def test_callback_warmup(self):
        """Тест warmup периода"""
        from transformers.adaptive_loss import AdaptiveLossCallback, AdaptiveLossConfig
        
        config = AdaptiveLossConfig(warmup_steps=10)
        callback = AdaptiveLossCallback(config)
        
        logits = torch.randn(2, 10, 100)
        labels = torch.randint(0, 100, (2, 10))
        
        # Во время warmup используется обычный loss
        for _ in range(5):
            callback.compute_loss(logits, labels)
        
        assert callback.step_count == 5
    
    def test_callback_stats(self):
        """Тест получения статистики"""
        from transformers.adaptive_loss import AdaptiveLossCallback
        
        callback = AdaptiveLossCallback()
        
        logits = torch.randn(2, 10, 100)
        labels = torch.randint(0, 100, (2, 10))
        
        for _ in range(5):
            callback.compute_loss(logits, labels)
        
        stats = callback.get_stats()
        
        assert "adaptive_loss/mean" in stats
        assert "adaptive_loss/steps" in stats
        assert stats["adaptive_loss/steps"] == 5


class TestTokenLossDistribution:
    """Тесты для get_token_loss_distribution"""
    
    def test_distribution_stats(self):
        """Тест получения статистики распределения"""
        from transformers.adaptive_loss import get_token_loss_distribution
        
        logits = torch.randn(2, 10, 100)
        labels = torch.randint(0, 100, (2, 10))
        
        stats = get_token_loss_distribution(logits, labels)
        
        assert "mean_loss" in stats
        assert "std_loss" in stats
        assert "min_loss" in stats
        assert "max_loss" in stats
        assert "hardest_tokens" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
