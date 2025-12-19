"""
Тесты для LR Finder v1.1.1
==========================

Запуск:
    pytest tests/forge/test_lr_finder.py -v
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass


class TestLRFinderResult:
    """Тесты для LRFinderResult dataclass"""
    
    def test_result_creation(self):
        """Тест создания LRFinderResult"""
        from transformers.lr_finder import LRFinderResult
        
        result = LRFinderResult(
            optimal_lr=2e-5,
            min_lr=1e-8,
            max_lr=1e-1,
            num_steps=100,
            lrs=[1e-7, 1e-6, 1e-5],
            losses=[2.0, 1.5, 1.2],
            smoothed_losses=[2.0, 1.6, 1.3],
        )
        
        assert result.optimal_lr == 2e-5
        assert result.num_steps == 100
        assert len(result.lrs) == 3
    
    def test_result_repr(self):
        """Тест строкового представления"""
        from transformers.lr_finder import LRFinderResult
        
        result = LRFinderResult(
            optimal_lr=2e-5,
            min_lr=1e-8,
            max_lr=1e-1,
            num_steps=50,
        )
        
        repr_str = repr(result)
        assert "LRFinderResult" in repr_str
        assert "2.00e-05" in repr_str


class TestLRFinderImport:
    """Тесты импорта LR Finder"""
    
    def test_import_from_module(self):
        """Тест импорта из модуля"""
        from transformers.lr_finder import LRFinder, LRFinderResult, find_optimal_lr
        
        assert LRFinder is not None
        assert LRFinderResult is not None
        assert find_optimal_lr is not None
    
    def test_import_from_main(self):
        """Тест импорта из главного модуля"""
        from transformers import LRFinder, find_optimal_lr
        
        assert LRFinder is not None
        assert find_optimal_lr is not None


class TestLRFinderInit:
    """Тесты инициализации LRFinder"""
    
    @pytest.fixture
    def mock_model(self):
        """Мок модели PyTorch"""
        try:
            import torch
            import torch.nn as nn
            
            class SimpleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(10, 10)
                
                def forward(self, input_ids, labels=None, **kwargs):
                    output = self.linear(input_ids.float())
                    loss = output.mean()
                    return MagicMock(loss=loss)
            
            return SimpleModel()
        except ImportError:
            pytest.skip("PyTorch не установлен")
    
    @pytest.fixture
    def mock_dataloader(self):
        """Мок DataLoader"""
        try:
            import torch
            from torch.utils.data import DataLoader, TensorDataset
            
            data = torch.randn(32, 10)
            labels = torch.randn(32, 10)
            dataset = TensorDataset(data, labels)
            
            def collate_fn(batch):
                inputs, targets = zip(*batch)
                return {
                    "input_ids": torch.stack(inputs),
                    "labels": torch.stack(targets)
                }
            
            return DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
        except ImportError:
            pytest.skip("PyTorch не установлен")
    
    def test_lr_finder_init(self, mock_model, mock_dataloader):
        """Тест инициализации LRFinder"""
        from transformers.lr_finder import LRFinder
        
        finder = LRFinder(mock_model, mock_dataloader, device="cpu")
        
        assert finder.model is mock_model
        assert finder.train_dataloader is mock_dataloader
        assert finder.result is None


class TestLRFinderMethods:
    """Тесты методов LRFinder"""
    
    def test_smooth_losses(self):
        """Тест сглаживания loss"""
        from transformers.lr_finder import LRFinder
        
        # Создаём мок LRFinder без реальных зависимостей
        with patch.object(LRFinder, '__init__', lambda x, *args, **kwargs: None):
            finder = LRFinder.__new__(LRFinder)
            
            losses = [2.0, 1.8, 1.5, 1.2, 1.0]
            smoothed = finder._smooth_losses(losses, beta=0.9)
            
            assert len(smoothed) == len(losses)
            # Сглаженные значения должны быть менее volatile
            assert all(isinstance(s, float) for s in smoothed)
    
    def test_find_steep_gradient(self):
        """Тест поиска точки максимального градиента"""
        from transformers.lr_finder import LRFinder
        import math
        
        with patch.object(LRFinder, '__init__', lambda x, *args, **kwargs: None):
            finder = LRFinder.__new__(LRFinder)
            
            lrs = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
            losses = [2.0, 1.5, 0.8, 1.0, 2.0]  # Минимум в середине
            
            idx = finder._find_steep_gradient(lrs, losses)
            
            # Индекс должен быть близко к точке максимального спуска
            assert 0 <= idx < len(lrs) - 1


class TestLRFinderRestore:
    """Тесты восстановления модели"""
    
    def test_model_restored_after_find(self):
        """Тест что веса восстанавливаются после find()"""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            pytest.skip("PyTorch не установлен")
        
        from transformers.lr_finder import LRFinder
        from torch.utils.data import DataLoader, TensorDataset
        
        # Простая модель
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
            
            def forward(self, input_ids, labels=None, **kwargs):
                output = self.linear(input_ids.float())
                loss = output.mean()
                return MagicMock(loss=loss)
        
        model = SimpleModel()
        
        # Сохраняем оригинальные веса
        original_weights = model.linear.weight.data.clone()
        
        # Создаём данные
        data = torch.randn(32, 10)
        labels = torch.randn(32, 10)
        dataset = TensorDataset(data, labels)
        
        def collate_fn(batch):
            inputs, targets = zip(*batch)
            return {
                "input_ids": torch.stack(inputs),
                "labels": torch.stack(targets)
            }
        
        dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
        
        # LR Finder
        finder = LRFinder(model, dataloader, device="cpu")
        finder.find(num_steps=10)  # Короткий тест
        
        # Проверяем что веса восстановлены
        restored_weights = model.linear.weight.data
        assert torch.allclose(original_weights, restored_weights)


class TestLRFinderResult:
    """Тесты результатов LRFinder"""
    
    def test_result_has_all_fields(self):
        """Тест что результат содержит все необходимые поля"""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            pytest.skip("PyTorch не установлен")
        
        from transformers.lr_finder import LRFinder
        from torch.utils.data import DataLoader, TensorDataset
        
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
            
            def forward(self, input_ids, labels=None, **kwargs):
                output = self.linear(input_ids.float())
                loss = output.mean()
                return MagicMock(loss=loss)
        
        model = SimpleModel()
        data = torch.randn(32, 10)
        labels = torch.randn(32, 10)
        dataset = TensorDataset(data, labels)
        
        def collate_fn(batch):
            inputs, targets = zip(*batch)
            return {"input_ids": torch.stack(inputs), "labels": torch.stack(targets)}
        
        dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
        finder = LRFinder(model, dataloader, device="cpu")
        
        result = finder.find(num_steps=10)
        
        # Проверяем все поля
        assert result.optimal_lr is not None
        assert result.optimal_lr > 0
        assert result.min_lr == 1e-8
        assert result.max_lr == 1e-1
        assert len(result.lrs) > 0
        assert len(result.losses) > 0
        assert len(result.smoothed_losses) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
