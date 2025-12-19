"""
Тесты для Interactive Model Manager v1.0.9
==========================================

Запуск:
    pytest tests/forge/test_interactive.py -v
"""

import json
import os
import tempfile
from pathlib import Path

import pytest


class TestModelInfo:
    """Тесты для ModelInfo dataclass"""
    
    def test_model_info_creation(self):
        """Тест создания ModelInfo"""
        from transformers.interactive import ModelInfo
        
        info = ModelInfo(
            name="TestModel",
            path="/path/to/model",
            size_gb=5.5,
            model_type="LlamaForCausalLM"
        )
        
        assert info.name == "TestModel"
        assert info.size_gb == 5.5
        assert info.model_type == "LlamaForCausalLM"


class TestDatasetInfo:
    """Тесты для DatasetInfo dataclass"""
    
    def test_dataset_info_creation(self):
        """Тест создания DatasetInfo"""
        from transformers.interactive import DatasetInfo
        
        info = DatasetInfo(
            name="train.jsonl",
            path="/path/to/train.jsonl",
            size_mb=100.5,
            num_lines=5000,
            format="jsonl"
        )
        
        assert info.name == "train.jsonl"
        assert info.num_lines == 5000
        assert info.format == "jsonl"


class TestValidationResult:
    """Тесты для ValidationResult"""
    
    def test_valid_result(self):
        """Тест валидного результата"""
        from transformers.interactive import ValidationResult
        
        result = ValidationResult(
            valid=True,
            total_lines=100,
            valid_lines=100
        )
        
        assert result.valid == True
        assert result.total_lines == 100
    
    def test_invalid_result(self):
        """Тест невалидного результата"""
        from transformers.interactive import ValidationResult
        
        result = ValidationResult(
            valid=False,
            errors=["Line 1: Missing 'messages'"]
        )
        
        assert result.valid == False
        assert len(result.errors) == 1


class TestInteractiveModelManager:
    """Тесты для InteractiveModelManager"""
    
    def test_manager_init(self):
        """Тест инициализации менеджера"""
        from transformers.interactive import InteractiveModelManager
        
        manager = InteractiveModelManager(
            models_dir="./test_models",
            datasets_dir="./test_datasets"
        )
        
        assert str(manager.models_dir) == "test_models"
        assert str(manager.datasets_dir) == "test_datasets"
    
    def test_format_params(self):
        """Тест форматирования параметров"""
        from transformers.interactive import InteractiveModelManager
        
        manager = InteractiveModelManager()
        
        assert manager._format_params(7_000_000_000) == "7.0B"
        assert manager._format_params(125_000_000) == "125.0M"
        assert manager._format_params(50_000) == "50.0K"
    
    def test_scan_empty_dir(self):
        """Тест сканирования пустой папки"""
        from transformers.interactive import InteractiveModelManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = InteractiveModelManager(models_dir=tmpdir)
            models = manager.scan_models()
            
            assert len(models) == 0
    
    def test_scan_no_dir(self):
        """Тест сканирования несуществующей папки"""
        from transformers.interactive import InteractiveModelManager
        
        manager = InteractiveModelManager(models_dir="/nonexistent/path")
        models = manager.scan_models()
        
        assert len(models) == 0


class TestDatasetValidation:
    """Тесты валидации датасетов"""
    
    def test_validate_valid_dataset(self):
        """Тест валидации правильного датасета"""
        from transformers.interactive import InteractiveModelManager
        
        manager = InteractiveModelManager()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            # Valid ChatML format
            data = {"messages": [
                {"role": "system", "content": "Test"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"}
            ]}
            f.write(json.dumps(data) + "\n")
            f.write(json.dumps(data) + "\n")
            f.flush()
            
            result = manager.validate_dataset(f.name)
            
            assert result.valid == True
            assert result.total_lines == 2
            assert result.valid_lines == 2
        
        os.unlink(f.name)
    
    def test_validate_missing_messages(self):
        """Тест валидации датасета без messages"""
        from transformers.interactive import InteractiveModelManager
        
        manager = InteractiveModelManager()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            # Invalid - no messages field
            data = {"text": "Hello world"}
            f.write(json.dumps(data) + "\n")
            f.flush()
            
            result = manager.validate_dataset(f.name)
            
            assert result.valid == False
            assert any("messages" in err.lower() for err in result.errors)
        
        os.unlink(f.name)
    
    def test_validate_missing_role(self):
        """Тест валидации сообщения без role"""
        from transformers.interactive import InteractiveModelManager
        
        manager = InteractiveModelManager()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            # Invalid - message without role
            data = {"messages": [{"content": "Hello"}]}
            f.write(json.dumps(data) + "\n")
            f.flush()
            
            result = manager.validate_dataset(f.name)
            
            assert result.valid == False
            assert any("role" in err.lower() for err in result.errors)
        
        os.unlink(f.name)
    
    def test_validate_invalid_json(self):
        """Тест валидации невалидного JSON"""
        from transformers.interactive import InteractiveModelManager
        
        manager = InteractiveModelManager()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write("not valid json\n")
            f.flush()
            
            result = manager.validate_dataset(f.name)
            
            assert result.valid == False
            assert any("json" in err.lower() for err in result.errors)
        
        os.unlink(f.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
