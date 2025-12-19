"""
Тесты для модуля Layer Utils
============================

Запуск:
    pytest tests/forge/test_layer_utils.py -v
"""

import pytest
import torch
import torch.nn as nn


class SimpleTransformerModel(nn.Module):
    """Простая модель для тестов, имитирующая структуру трансформера"""
    
    def __init__(self, num_layers=4, hidden_size=64):
        super().__init__()
        
        self.embed_tokens = nn.Embedding(1000, hidden_size)
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=4,
                dim_feedforward=hidden_size * 4,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        self.lm_head = nn.Linear(hidden_size, 1000)
    
    def forward(self, x):
        x = self.embed_tokens(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


class TestFreezeFunctions:
    """Тесты для функций заморозки"""
    
    @pytest.fixture
    def model(self):
        """Создаём тестовую модель"""
        return SimpleTransformerModel(num_layers=4)
    
    def test_freeze_model(self, model):
        """Тест freeze_model"""
        from transformers.layer_utils import freeze_model
        
        # Проверяем что изначально всё размороженно
        assert all(p.requires_grad for p in model.parameters())
        
        # Замораживаем
        frozen_count = freeze_model(model)
        
        # Проверяем что всё заморожено
        assert frozen_count > 0
        assert all(not p.requires_grad for p in model.parameters())
    
    def test_unfreeze_model(self, model):
        """Тест unfreeze_model"""
        from transformers.layer_utils import freeze_model, unfreeze_model
        
        # Замораживаем сначала
        freeze_model(model)
        assert all(not p.requires_grad for p in model.parameters())
        
        # Размораживаем
        unfrozen_count = unfreeze_model(model)
        
        # Проверяем что всё разморожено
        assert unfrozen_count > 0
        assert all(p.requires_grad for p in model.parameters())
    
    def test_freeze_first_n_layers(self, model):
        """Тест freeze_first_n_layers"""
        from transformers.layer_utils import freeze_first_n_layers
        
        # Замораживаем первые 2 слоя
        freeze_first_n_layers(model, n=2)
        
        # Проверяем что первые 2 слоя заморожены
        for i, layer in enumerate(model.layers):
            if i < 2:
                assert all(not p.requires_grad for p in layer.parameters()), f"Layer {i} should be frozen"
            else:
                assert all(p.requires_grad for p in layer.parameters()), f"Layer {i} should be trainable"
    
    def test_freeze_except_last_n(self, model):
        """Тест freeze_except_last_n"""
        from transformers.layer_utils import freeze_except_last_n
        
        # Замораживаем всё кроме последних 2 слоёв
        freeze_except_last_n(model, n=2)
        
        # Проверяем
        num_layers = len(model.layers)
        for i, layer in enumerate(model.layers):
            if i >= num_layers - 2:
                assert all(p.requires_grad for p in layer.parameters()), f"Layer {i} should be trainable"
    
    def test_freeze_embeddings(self, model):
        """Тест freeze_embeddings"""
        from transformers.layer_utils import freeze_embeddings
        
        freeze_embeddings(model)
        
        # Проверяем что эмбеддинги заморожены
        assert all(not p.requires_grad for p in model.embed_tokens.parameters())


class TestAnalysisFunctions:
    """Тесты для функций анализа"""
    
    @pytest.fixture
    def model(self):
        """Создаём тестовую модель"""
        return SimpleTransformerModel(num_layers=4)
    
    def test_get_trainable_params(self, model):
        """Тест get_trainable_params"""
        from transformers.layer_utils import get_trainable_params, freeze_model
        
        total = get_trainable_params(model)
        assert total > 0
        
        # После заморозки должно быть 0
        freeze_model(model)
        trainable = get_trainable_params(model)
        assert trainable == 0
    
    def test_get_frozen_percentage(self, model):
        """Тест get_frozen_percentage"""
        from transformers.layer_utils import get_frozen_percentage, freeze_model
        
        # Изначально 0%
        pct = get_frozen_percentage(model)
        assert pct == 0.0
        
        # После заморозки 100%
        freeze_model(model)
        pct = get_frozen_percentage(model)
        assert pct == 100.0
    
    def test_get_num_layers(self, model):
        """Тест get_num_layers"""
        from transformers.layer_utils import get_num_layers
        
        num = get_num_layers(model)
        assert num == 4
    
    def test_print_layer_status(self, model, capsys):
        """Тест print_layer_status"""
        from transformers.layer_utils import print_layer_status
        
        print_layer_status(model)
        
        captured = capsys.readouterr()
        # Должен содержать какой-то вывод
        assert len(captured.out) > 0


class TestLPLoRAStyle:
    """Тесты для LP-LoRA стиля"""
    
    @pytest.fixture
    def model(self):
        """Создаём тестовую модель"""
        return SimpleTransformerModel(num_layers=4)
    
    def test_setup_lp_lora_style(self, model):
        """Тест setup_lp_lora_style"""
        from transformers.layer_utils import setup_lp_lora_style, get_frozen_percentage
        
        setup_lp_lora_style(model, freeze_ratio=0.5)
        
        pct = get_frozen_percentage(model)
        # Должно быть примерно 50% заморожено
        assert 30 < pct < 70
    
    def test_get_memory_savings_estimate(self, model):
        """Тест get_memory_savings_estimate"""
        from transformers.layer_utils import (
            get_memory_savings_estimate, 
            freeze_first_n_layers
        )
        
        # Замораживаем половину
        freeze_first_n_layers(model, n=2)
        
        savings = get_memory_savings_estimate(model)
        
        assert "gradient_saved_gb" in savings
        assert "optimizer_saved_gb" in savings
        assert "total_saved_gb" in savings
        assert savings["total_saved_gb"] >= 0


class TestGradualUnfreezer:
    """Тесты для GradualUnfreezer"""
    
    @pytest.fixture
    def model(self):
        """Создаём тестовую модель"""
        return SimpleTransformerModel(num_layers=4)
    
    def test_gradual_unfreezer_init(self, model):
        """Тест инициализации"""
        from transformers.layer_utils import GradualUnfreezer
        
        unfreezer = GradualUnfreezer(model, total_epochs=4, verbose=False)
        
        # После инициализации модель должна быть заморожена
        from transformers.layer_utils import get_frozen_percentage
        pct = get_frozen_percentage(model)
        assert pct > 50  # Большая часть заморожена
    
    def test_gradual_unfreezer_step(self, model):
        """Тест постепенной разморозки"""
        from transformers.layer_utils import GradualUnfreezer, get_frozen_percentage
        
        unfreezer = GradualUnfreezer(model, total_epochs=4, verbose=False)
        
        frozen_percentages = []
        
        for epoch in range(4):
            unfreezer.step(epoch)
            pct = get_frozen_percentage(model)
            frozen_percentages.append(pct)
        
        # Процент замороженного должен уменьшаться
        # (или оставаться стабильным если уже разморожено)
        assert frozen_percentages[-1] <= frozen_percentages[0]


class TestFreezeByName:
    """Тесты для freeze_by_name"""
    
    @pytest.fixture
    def model(self):
        """Создаём тестовую модель"""
        return SimpleTransformerModel(num_layers=4)
    
    def test_freeze_by_name_single_pattern(self, model):
        """Тест заморозки по одному паттерну"""
        from transformers.layer_utils import freeze_by_name
        
        # Замораживаем только lm_head
        freeze_by_name(model, "lm_head")
        
        # Проверяем что lm_head заморожен
        assert all(not p.requires_grad for p in model.lm_head.parameters())
        
        # Остальные не заморожены
        assert any(p.requires_grad for p in model.layers.parameters())
    
    def test_freeze_by_name_multiple_patterns(self, model):
        """Тест заморозки по нескольким паттернам"""
        from transformers.layer_utils import freeze_by_name
        
        # Замораживаем embeddings и lm_head
        freeze_by_name(model, ["embed", "lm_head"])
        
        # Проверяем
        assert all(not p.requires_grad for p in model.embed_tokens.parameters())
        assert all(not p.requires_grad for p in model.lm_head.parameters())

class TestNewUtilities:
    """Тесты для новых утилит v1.0.7"""
    
    @pytest.fixture
    def model(self):
        """Создаём тестовую модель"""
        return SimpleTransformerModel(num_layers=4)
    
    def test_get_layer_names(self, model):
        """Тест get_layer_names"""
        from transformers.layer_utils import get_layer_names
        
        # Без параметров
        names = get_layer_names(model)
        assert isinstance(names, list)
        assert len(names) > 0
        
        # С параметрами
        names_with_params = get_layer_names(model, include_params=True)
        assert len(names_with_params) > len(names)
    
    def test_estimate_training_time(self, model):
        """Тест estimate_training_time"""
        from transformers.layer_utils import estimate_training_time
        
        estimate = estimate_training_time(
            model=model,
            num_samples=10000,
            batch_size=8,
            num_epochs=3
        )
        
        assert "total_steps" in estimate
        assert "formatted" in estimate
        assert estimate["total_steps"] > 0
        assert "h" in estimate["formatted"]
    
    def test_print_model_summary(self, model, capsys):
        """Тест print_model_summary"""
        from transformers.layer_utils import print_model_summary
        
        # Вызываем функцию
        print_model_summary(model)
        
        # Проверяем что был вывод
        captured = capsys.readouterr()
        assert "MODEL SUMMARY" in captured.out
        assert "Parameters" in captured.out
        assert "Trainable" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
