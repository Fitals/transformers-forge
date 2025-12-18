"""
Тесты для модуля EMA (Exponential Moving Average)
=================================================

Запуск:
    pytest tests/forge/test_ema.py -v
"""

import pytest
import torch
import torch.nn as nn


class TestEMAFunctions:
    """Тесты для функций EMA"""
    
    @pytest.fixture
    def simple_model(self):
        """Простая модель для тестов"""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        return model
    
    def test_create_ema_state(self, simple_model):
        """Тест создания EMA state"""
        from transformers.ema import create_ema_state
        
        ema_state = create_ema_state(simple_model)
        
        # Проверяем что state создан
        assert ema_state is not None
        assert isinstance(ema_state, dict)
        assert len(ema_state) > 0
    
    def test_update_ema_state(self, simple_model):
        """Тест обновления EMA state"""
        from transformers.ema import create_ema_state, update_ema_state
        
        ema_state = create_ema_state(simple_model)
        original_ema = {k: v.clone() for k, v in ema_state.items()}
        
        # Изменяем веса модели
        for param in simple_model.parameters():
            param.data.add_(torch.randn_like(param.data) * 0.1)
        
        # Обновляем EMA
        decay = 0.9
        update_ema_state(simple_model, ema_state, decay=decay)
        
        # Проверяем что EMA изменился
        for name, ema_param in ema_state.items():
            assert not torch.allclose(ema_param, original_ema[name])
    
    def test_apply_ema_state(self, simple_model):
        """Тест применения EMA state к модели"""
        from transformers.ema import create_ema_state, apply_ema_state
        
        # Создаём EMA state
        ema_state = create_ema_state(simple_model)
        
        # Изменяем EMA state (симулируем обучение)
        for name in ema_state:
            ema_state[name] = torch.randn_like(ema_state[name])
        
        # Применяем EMA
        backup = apply_ema_state(simple_model, ema_state)
        
        # Проверяем что backup создан
        assert backup is not None
        assert isinstance(backup, dict)
    
    def test_ema_state_dict_save_load(self, simple_model):
        """Тест сохранения и загрузки EMA state dict"""
        from transformers.ema import create_ema_state, get_ema_state_dict, load_ema_state_dict
        
        ema_state = create_ema_state(simple_model)
        
        # Сохраняем
        state_dict = get_ema_state_dict(ema_state)
        
        # Загружаем
        new_ema_state = load_ema_state_dict(state_dict)
        
        # Проверяем что загрузилось корректно
        assert len(new_ema_state) == len(state_dict)


class TestEMACallback:
    """Тесты для EMACallback"""
    
    @pytest.fixture
    def simple_model(self):
        """Простая модель"""
        return nn.Linear(10, 5)
    
    def test_callback_init(self):
        """Тест инициализации callback"""
        from transformers.ema import EMACallback
        
        callback = EMACallback(decay=0.999)
        
        assert callback.decay == 0.999
        assert callback.ema_state is None
    
    def test_callback_init_with_params(self):
        """Тест инициализации с параметрами"""
        from transformers.ema import EMACallback
        
        callback = EMACallback(
            decay=0.99,
            update_after_step=100,
            update_every=2
        )
        
        assert callback.decay == 0.99
        assert callback.update_after_step == 100
        assert callback.update_every == 2


class TestEMAModel:
    """Тесты для EMAModel"""
    
    @pytest.fixture
    def simple_model(self):
        """Простая модель"""
        return nn.Linear(10, 5)
    
    def test_ema_model_init(self, simple_model):
        """Тест инициализации EMAModel"""
        from transformers.ema import EMAModel
        
        ema_model = EMAModel(simple_model, decay=0.999)
        
        assert ema_model.model is simple_model
        assert ema_model.decay == 0.999
        assert ema_model.ema_state is not None
    
    def test_ema_model_update(self, simple_model):
        """Тест обновления EMA"""
        from transformers.ema import EMAModel
        
        ema_model = EMAModel(simple_model, decay=0.9)
        original_ema = {k: v.clone() for k, v in ema_model.ema_state.items()}
        
        # Изменяем веса модели
        for param in simple_model.parameters():
            param.data.add_(torch.randn_like(param.data) * 0.1)
        
        # Обновляем
        ema_model.update()
        
        # Проверяем что EMA изменился
        for name, ema_param in ema_model.ema_state.items():
            assert not torch.allclose(ema_param, original_ema[name])
    
    def test_ema_model_use_ema_context(self, simple_model):
        """Тест context manager use_ema"""
        from transformers.ema import EMAModel
        
        ema_model = EMAModel(simple_model, decay=0.999)
        
        # Изменяем EMA state
        for name in ema_model.ema_state:
            ema_model.ema_state[name] = torch.randn_like(ema_model.ema_state[name])
        
        # Сохраняем текущие веса
        current_weights = {
            name: param.clone() 
            for name, param in simple_model.named_parameters()
        }
        
        # Используем EMA в context
        with ema_model.use_ema():
            # Внутри должны быть EMA веса
            pass
        
        # После выхода должны восстановиться оригинальные
        for name, param in simple_model.named_parameters():
            assert torch.allclose(param.data, current_weights[name])


class TestEMAUtils:
    """Тесты для утилит EMA"""
    
    def test_compute_optimal_decay(self):
        """Тест расчёта оптимального decay"""
        from transformers.ema import compute_optimal_decay
        
        # Короткое обучение -> низкий decay
        decay_short = compute_optimal_decay(1000)
        
        # Длинное обучение -> высокий decay
        decay_long = compute_optimal_decay(100000)
        
        assert decay_short < decay_long
        assert 0.9 < decay_short < 1.0
        assert 0.99 < decay_long < 1.0
    
    def test_print_ema_info(self, capsys):
        """Тест вывода информации о EMA"""
        from transformers.ema import print_ema_info
        
        print_ema_info(decay=0.999, total_steps=10000)
        
        captured = capsys.readouterr()
        assert "0.999" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
