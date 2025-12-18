"""
Тесты для модуля Training Monitor
=================================

Запуск:
    pytest tests/forge/test_training_monitor.py -v
"""

import pytest
import torch
import torch.nn as nn


class TestCountParameters:
    """Тесты для count_parameters"""
    
    @pytest.fixture
    def model(self):
        """Простая модель"""
        return nn.Linear(100, 50)
    
    def test_count_all_parameters(self, model):
        """Тест подсчёта всех параметров"""
        from transformers.training_monitor import count_parameters
        
        total = count_parameters(model)
        
        # Linear(100, 50) = 100*50 + 50 = 5050
        assert total == 5050
    
    def test_count_trainable_only(self, model):
        """Тест подсчёта только обучаемых параметров"""
        from transformers.training_monitor import count_parameters
        
        # Замораживаем bias
        model.bias.requires_grad = False
        
        trainable = count_parameters(model, trainable_only=True)
        total = count_parameters(model, trainable_only=False)
        
        assert trainable < total
        # trainable = 100*50 = 5000
        assert trainable == 5000


class TestFormatParamCount:
    """Тесты для format_param_count"""
    
    def test_format_billions(self):
        """Тест форматирования миллиардов"""
        from transformers.training_monitor import format_param_count
        
        result = format_param_count(1_500_000_000)
        assert result == "1.50B"
    
    def test_format_millions(self):
        """Тест форматирования миллионов"""
        from transformers.training_monitor import format_param_count
        
        result = format_param_count(7_000_000)
        assert result == "7.00M"
    
    def test_format_thousands(self):
        """Тест форматирования тысяч"""
        from transformers.training_monitor import format_param_count
        
        result = format_param_count(5_000)
        assert result == "5.00K"
    
    def test_format_small(self):
        """Тест форматирования малых чисел"""
        from transformers.training_monitor import format_param_count
        
        result = format_param_count(500)
        assert "500" in result


class TestGetParameterBreakdown:
    """Тесты для get_parameter_breakdown"""
    
    @pytest.fixture
    def model(self):
        """Простая модель"""
        return nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
    
    def test_breakdown_returns_list(self, model):
        """Тест что возвращается список"""
        from transformers.training_monitor import get_parameter_breakdown
        
        breakdown = get_parameter_breakdown(model)
        
        assert isinstance(breakdown, list)
        assert len(breakdown) > 0
    
    def test_breakdown_has_required_fields(self, model):
        """Тест наличия обязательных полей"""
        from transformers.training_monitor import get_parameter_breakdown
        
        breakdown = get_parameter_breakdown(model)
        
        for item in breakdown:
            assert "name" in item
            assert "params" in item


class TestEstimateModelMemory:
    """Тесты для estimate_model_memory"""
    
    @pytest.fixture
    def model(self):
        """Простая модель"""
        return nn.Linear(1000, 500)
    
    def test_estimate_returns_dict(self, model):
        """Тест что возвращается словарь"""
        from transformers.training_monitor import estimate_model_memory
        
        memory = estimate_model_memory(model)
        
        assert isinstance(memory, dict)
    
    def test_estimate_has_required_fields(self, model):
        """Тест наличия обязательных полей"""
        from transformers.training_monitor import estimate_model_memory
        
        memory = estimate_model_memory(model)
        
        assert "parameters_gb" in memory or "total_estimated_gb" in memory
    
    def test_estimate_positive_values(self, model):
        """Тест что значения положительные"""
        from transformers.training_monitor import estimate_model_memory
        
        memory = estimate_model_memory(model)
        
        for key, value in memory.items():
            if isinstance(value, (int, float)):
                assert value >= 0


class TestPrintModelInfo:
    """Тесты для print_model_info"""
    
    @pytest.fixture
    def model(self):
        """Простая модель"""
        return nn.Linear(100, 50)
    
    def test_print_model_info(self, model, capsys):
        """Тест вывода информации"""
        from transformers.training_monitor import print_model_info
        
        print_model_info(model)
        
        captured = capsys.readouterr()
        assert len(captured.out) > 0
        # Должен содержать информацию о параметрах
        assert "parameter" in captured.out.lower() or "param" in captured.out.lower()


class TestCheckGradientHealth:
    """Тесты для check_gradient_health"""
    
    @pytest.fixture
    def model_with_grads(self):
        """Модель с градиентами"""
        model = nn.Linear(10, 5)
        
        # Forward и backward
        x = torch.randn(2, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        
        return model
    
    def test_check_healthy_gradients(self, model_with_grads):
        """Тест проверки здоровых градиентов"""
        from transformers.training_monitor import check_gradient_health
        
        health = check_gradient_health(model_with_grads)
        
        assert isinstance(health, dict)
        assert "healthy" in health
        assert health["healthy"] == True
    
    def test_check_gradient_with_nan(self):
        """Тест обнаружения NaN градиентов"""
        from transformers.training_monitor import check_gradient_health
        
        model = nn.Linear(10, 5)
        
        # Создаём NaN градиент
        for p in model.parameters():
            p.grad = torch.full_like(p, float('nan'))
        
        health = check_gradient_health(model)
        
        assert health["healthy"] == False
    
    def test_check_gradient_with_inf(self):
        """Тест обнаружения Inf градиентов"""
        from transformers.training_monitor import check_gradient_health
        
        model = nn.Linear(10, 5)
        
        # Создаём Inf градиент
        for p in model.parameters():
            p.grad = torch.full_like(p, float('inf'))
        
        health = check_gradient_health(model)
        
        assert health["healthy"] == False


class TestTrainingMonitor:
    """Тесты для TrainingMonitor"""
    
    @pytest.fixture
    def model(self):
        """Простая модель"""
        return nn.Linear(100, 50)
    
    def test_monitor_init(self, model):
        """Тест инициализации монитора"""
        from transformers.training_monitor import TrainingMonitor
        
        monitor = TrainingMonitor(model)
        
        assert monitor.model is model
    
    def test_get_model_summary(self, model):
        """Тест получения summary"""
        from transformers.training_monitor import TrainingMonitor
        
        monitor = TrainingMonitor(model)
        summary = monitor.get_model_summary()
        
        assert isinstance(summary, dict)
        assert "total_parameters" in summary
        assert "trainable_parameters" in summary


class TestMonitorCallback:
    """Тесты для MonitorCallback"""
    
    def test_callback_init(self):
        """Тест инициализации callback"""
        from transformers.training_monitor import MonitorCallback
        
        callback = MonitorCallback(
            print_model_summary=True,
            log_gpu_memory=False,
            check_gradients=True
        )
        
        assert callback.print_model_summary == True
        assert callback.log_gpu_memory == False
        assert callback.check_gradients == True
    
    def test_callback_default_init(self):
        """Тест дефолтной инициализации"""
        from transformers.training_monitor import MonitorCallback
        
        callback = MonitorCallback()
        
        # Не должен падать при создании
        assert callback is not None


class TestGradientStats:
    """Тесты для GradientStats"""
    
    def test_gradient_stats_dataclass(self):
        """Тест что GradientStats это dataclass"""
        from transformers.training_monitor import GradientStats
        
        stats = GradientStats(
            min_grad=0.0,
            max_grad=1.0,
            mean_grad=0.5,
            std_grad=0.2,
            num_zero=0,
            num_nan=0,
            num_inf=0
        )
        
        assert stats.min_grad == 0.0
        assert stats.max_grad == 1.0
        assert stats.mean_grad == 0.5


class TestTrainingMetrics:
    """Тесты для TrainingMetrics"""
    
    def test_training_metrics_dataclass(self):
        """Тест что TrainingMetrics это dataclass"""
        from transformers.training_monitor import TrainingMetrics
        
        metrics = TrainingMetrics(
            step=100,
            loss=0.5,
            learning_rate=1e-4,
            gradient_norm=1.0,
            gpu_memory_used=5.0,
            gpu_memory_total=24.0
        )
        
        assert metrics.step == 100
        assert metrics.loss == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
