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
    
    def test_breakdown_returns_dict(self, model):
        """Тест что возвращается словарь"""
        from transformers.training_monitor import get_parameter_breakdown
        
        breakdown = get_parameter_breakdown(model)
        
        assert isinstance(breakdown, dict)
        assert len(breakdown) > 0
    
    def test_breakdown_has_values(self, model):
        """Тест наличия данных в breakdown"""
        from transformers.training_monitor import get_parameter_breakdown
        
        breakdown = get_parameter_breakdown(model)
        
        # Проверяем что есть записи
        for key, info in breakdown.items():
            assert "total" in info
            assert "trainable" in info


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
            log_gradient_health=True
        )
        
        assert callback.print_model_summary == True
        assert callback.log_gpu_memory == False
        assert callback.log_gradient_health == True
    
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
        
        # Используем правильные имена полей
        stats = GradientStats(
            min_val=0.0,
            max_val=1.0,
            mean=0.5,
            std=0.2,
            norm=1.0,
            num_zeros=0,
            num_nans=0,
            num_infs=0
        )
        
        assert stats.min_val == 0.0
        assert stats.max_val == 1.0
        assert stats.mean == 0.5


class TestTrainingMetrics:
    """Тесты для TrainingMetrics"""
    
    def test_training_metrics_dataclass(self):
        """Тест что TrainingMetrics это dataclass"""
        from transformers.training_monitor import TrainingMetrics
        
        # Используем правильную структуру
        metrics = TrainingMetrics()
        
        assert hasattr(metrics, 'losses')
        assert hasattr(metrics, 'steps')
        assert isinstance(metrics.losses, list)

class TestProgressCallback:
    """Тесты для ProgressCallback"""
    
    def test_callback_init(self):
        """Тест инициализации ProgressCallback"""
        from transformers.training_monitor import ProgressCallback
        
        callback = ProgressCallback(
            show_eta=True,
            show_gpu=False,
            show_loss=True,
            update_every=5
        )
        
        assert callback.show_eta == True
        assert callback.show_gpu == False
        assert callback.show_loss == True
        assert callback.update_every == 5
    
    def test_callback_default_init(self):
        """Тест дефолтной инициализации"""
        from transformers.training_monitor import ProgressCallback
        
        callback = ProgressCallback()
        
        assert callback.show_eta == True
        assert callback.use_unicode == True
        assert callback.bar_width == 25
    
    def test_format_time_seconds(self):
        """Тест форматирования времени в секундах"""
        from transformers.training_monitor import ProgressCallback
        
        callback = ProgressCallback()
        
        result = callback._format_time(45)
        assert "45s" in result
    
    def test_format_time_minutes(self):
        """Тест форматирования времени в минутах"""
        from transformers.training_monitor import ProgressCallback
        
        callback = ProgressCallback()
        
        result = callback._format_time(125)  # 2m 5s
        assert "2m" in result
    
    def test_format_time_hours(self):
        """Тест форматирования времени в часах"""
        from transformers.training_monitor import ProgressCallback
        
        callback = ProgressCallback()
        
        result = callback._format_time(3700)  # 1h 1m
        assert "1h" in result
    
    def test_make_progress_bar(self):
        """Тест создания прогресс-бара"""
        from transformers.training_monitor import ProgressCallback
        
        callback = ProgressCallback(bar_width=10, use_unicode=True)
        
        bar_0 = callback._make_progress_bar(0.0)
        bar_50 = callback._make_progress_bar(0.5)
        bar_100 = callback._make_progress_bar(1.0)
        
        assert len(bar_0) == 10
        assert len(bar_50) == 10
        assert len(bar_100) == 10
        
        # 50% должен содержать и заполненные и пустые
        assert "█" in bar_50
        assert "░" in bar_50
    
    def test_make_progress_bar_ascii(self):
        """Тест создания ASCII прогресс-бара"""
        from transformers.training_monitor import ProgressCallback
        
        callback = ProgressCallback(bar_width=10, use_unicode=False)
        
        bar_50 = callback._make_progress_bar(0.5)
        
        assert "#" in bar_50
        assert "-" in bar_50


class TestFormatEta:
    """Тесты для format_eta"""
    
    def test_format_eta_seconds(self):
        """Тест форматирования секунд"""
        from transformers.training_monitor import format_eta
        
        result = format_eta(30)
        assert "30s" in result
    
    def test_format_eta_minutes(self):
        """Тест форматирования минут"""
        from transformers.training_monitor import format_eta
        
        result = format_eta(90)
        assert "1m" in result
    
    def test_format_eta_hours(self):
        """Тест форматирования часов"""
        from transformers.training_monitor import format_eta
        
        result = format_eta(3661)
        assert "1h" in result
    
    def test_format_eta_negative(self):
        """Тест отрицательного времени"""
        from transformers.training_monitor import format_eta
        
        result = format_eta(-10)
        assert "..." in result

class TestSmartCallbacks:
    """Тесты для Smart Training Callbacks v1.0.7"""
    
    def test_early_stopping_init(self):
        """Тест инициализации EarlyStoppingCallback"""
        from transformers.training_monitor import EarlyStoppingCallback
        
        callback = EarlyStoppingCallback(patience=5, metric="eval_loss")
        
        assert callback.patience == 5
        assert callback.metric == "eval_loss"
        assert callback.mode == "min"
        assert callback.best_value is None
        assert callback.interactive == False  # default
    
    def test_early_stopping_interactive_mode(self):
        """Тест interactive режима"""
        from transformers.training_monitor import EarlyStoppingCallback
        
        callback = EarlyStoppingCallback(patience=3, interactive=True)
        
        assert callback.interactive == True
    
    def test_early_stopping_invalid_mode(self):
        """Тест неправильного mode"""
        from transformers.training_monitor import EarlyStoppingCallback
        
        with pytest.raises(ValueError):
            EarlyStoppingCallback(mode="invalid")
    
    def test_early_stopping_is_improvement_min(self):
        """Тест _is_improvement для mode=min"""
        from transformers.training_monitor import EarlyStoppingCallback
        
        callback = EarlyStoppingCallback(mode="min", min_delta=0.01)
        
        # Улучшение: current < best - min_delta
        assert callback._is_improvement(0.5, 0.6) == True
        assert callback._is_improvement(0.59, 0.6) == False  # не достаточно
    
    def test_early_stopping_is_improvement_max(self):
        """Тест _is_improvement для mode=max"""
        from transformers.training_monitor import EarlyStoppingCallback
        
        callback = EarlyStoppingCallback(mode="max", min_delta=0.01)
        
        # Улучшение: current > best + min_delta
        assert callback._is_improvement(0.7, 0.6) == True
        assert callback._is_improvement(0.61, 0.6) == False  # не достаточно
    
    def test_reduce_lr_init(self):
        """Тест инициализации ReduceLROnPlateauCallback"""
        from transformers.training_monitor import ReduceLROnPlateauCallback
        
        callback = ReduceLROnPlateauCallback(factor=0.5, patience=2)
        
        assert callback.factor == 0.5
        assert callback.patience == 2
        assert callback.min_lr == 1e-7
    
    def test_reduce_lr_invalid_factor(self):
        """Тест неправильного factor"""
        from transformers.training_monitor import ReduceLROnPlateauCallback
        
        with pytest.raises(ValueError):
            ReduceLROnPlateauCallback(factor=1.5)
        
        with pytest.raises(ValueError):
            ReduceLROnPlateauCallback(factor=0)
    
    def test_best_model_init(self):
        """Тест инициализации BestModelCallback"""
        from transformers.training_monitor import BestModelCallback
        
        callback = BestModelCallback(save_path="./best", metric="eval_accuracy", mode="max")
        
        assert callback.save_path == "./best"
        assert callback.metric == "eval_accuracy"
        assert callback.mode == "max"
        assert callback.best_value is None
    
    def test_best_model_is_improvement(self):
        """Тест _is_improvement для BestModelCallback"""
        from transformers.training_monitor import BestModelCallback
        
        # mode=min (loss)
        callback_min = BestModelCallback(mode="min")
        assert callback_min._is_improvement(0.3, 0.5) == True
        assert callback_min._is_improvement(0.6, 0.5) == False
        
        # mode=max (accuracy)
        callback_max = BestModelCallback(mode="max")
        assert callback_max._is_improvement(0.9, 0.8) == True
        assert callback_max._is_improvement(0.7, 0.8) == False


class TestTrainingReportCallback:
    """Тесты для TrainingReportCallback v1.0.8"""
    
    def test_report_callback_init(self):
        """Тест инициализации TrainingReportCallback"""
        from transformers.training_monitor import TrainingReportCallback
        
        callback = TrainingReportCallback(
            output_path="./my_report.md",
            interactive=False
        )
        
        assert callback.output_path == "./my_report.md"
        assert callback.interactive == False
        assert callback.include_config == True
    
    def test_validate_model_name_valid(self):
        """Тест валидации правильных имён"""
        from transformers.training_monitor import TrainingReportCallback
        
        callback = TrainingReportCallback()
        
        # Valid names
        valid, _ = callback._validate_model_name("Ivan-3B")
        assert valid == True
        
        valid, _ = callback._validate_model_name("MyModel_v2")
        assert valid == True
        
        valid, _ = callback._validate_model_name("Test123")
        assert valid == True
    
    def test_validate_model_name_invalid_cyrillic(self):
        """Тест валидации кириллицы (запрещена)"""
        from transformers.training_monitor import TrainingReportCallback
        
        callback = TrainingReportCallback()
        
        valid, error = callback._validate_model_name("Иван-3B")
        assert valid == False
        assert "a-z" in error.lower() or "разрешены" in error.lower()
    
    def test_validate_model_name_too_long(self):
        """Тест слишком длинного названия"""
        from transformers.training_monitor import TrainingReportCallback
        
        callback = TrainingReportCallback()
        
        long_name = "A" * 60
        valid, error = callback._validate_model_name(long_name)
        assert valid == False
        assert "длинное" in error.lower() or "50" in error
    
    def test_validate_model_name_too_short(self):
        """Тест слишком короткого названия"""
        from transformers.training_monitor import TrainingReportCallback
        
        callback = TrainingReportCallback()
        
        valid, error = callback._validate_model_name("A")
        assert valid == False
        assert "короткое" in error.lower() or "2" in error
    
    def test_format_duration(self):
        """Тест форматирования времени"""
        from transformers.training_monitor import TrainingReportCallback
        
        callback = TrainingReportCallback()
        
        assert callback._format_duration(30) == "30s"
        assert callback._format_duration(90) == "1m 30s"
        assert callback._format_duration(3661) == "1h 1m 1s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
