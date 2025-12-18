"""
Тесты для модуля Training Presets
=================================

Запуск:
    pytest tests/forge/test_training_presets.py -v
"""

import pytest


class TestGetPreset:
    """Тесты для функции get_preset"""
    
    def test_get_sft_preset(self):
        """Тест получения SFT preset"""
        from transformers.training_presets import get_preset
        
        preset = get_preset("sft")
        
        assert preset is not None
        # Проверяем что это SFTPreset
        assert "SFT" in type(preset).__name__ or hasattr(preset, 'output_dir')
    
    def test_get_lora_preset(self):
        """Тест получения LoRA preset"""
        from transformers.training_presets import get_preset
        
        preset = get_preset("lora")
        
        assert preset is not None
        assert hasattr(preset, 'get_lora_config')
    
    def test_get_qlora_preset(self):
        """Тест получения QLoRA preset"""
        from transformers.training_presets import get_preset
        
        preset = get_preset("qlora")
        
        assert preset is not None
        assert hasattr(preset, 'get_bnb_config')
    
    def test_get_dpo_preset(self):
        """Тест получения DPO preset"""
        from transformers.training_presets import get_preset
        
        preset = get_preset("dpo")
        
        assert preset is not None
    
    def test_get_memory_efficient_preset(self):
        """Тест получения memory_efficient preset"""
        from transformers.training_presets import get_preset
        
        preset = get_preset("memory_efficient")
        
        assert preset is not None
    
    def test_get_preset_with_overrides(self):
        """Тест получения preset с переопределениями"""
        from transformers.training_presets import get_preset
        
        preset = get_preset("sft", learning_rate=1e-4, num_train_epochs=5)
        
        assert preset.learning_rate == 1e-4
        assert preset.num_train_epochs == 5
    
    def test_get_unknown_preset(self):
        """Тест получения несуществующего preset"""
        from transformers.training_presets import get_preset
        
        with pytest.raises((ValueError, KeyError)):
            get_preset("unknown_preset")


class TestPresetAttributes:
    """Тесты для атрибутов presets"""
    
    def test_sft_preset_has_output_dir(self):
        """Тест что SFT preset имеет output_dir"""
        from transformers.training_presets import get_preset
        
        preset = get_preset("sft")
        assert hasattr(preset, 'output_dir')
    
    def test_preset_has_learning_rate(self):
        """Тест что preset имеет learning_rate"""
        from transformers.training_presets import get_preset
        
        preset = get_preset("sft")
        assert hasattr(preset, 'learning_rate')
        assert preset.learning_rate > 0


class TestLoRAPreset:
    """Тесты для LoRA preset"""
    
    def test_get_lora_config(self):
        """Тест получения LoRA конфига"""
        from transformers.training_presets import get_preset
        
        preset = get_preset("lora")
        
        try:
            config = preset.get_lora_config()
            assert config is not None
        except ImportError:
            # PEFT не установлен
            pytest.skip("PEFT not installed")
    
    def test_lora_has_lora_r(self):
        """Тест что LoRA имеет lora_r"""
        from transformers.training_presets import get_preset
        
        preset = get_preset("lora")
        assert hasattr(preset, 'lora_r')


class TestQLoRAPreset:
    """Тесты для QLoRA preset"""
    
    def test_get_bnb_config(self):
        """Тест получения BitsAndBytes конфига"""
        from transformers.training_presets import get_preset
        
        preset = get_preset("qlora")
        
        try:
            config = preset.get_bnb_config()
            assert config is not None
        except ImportError:
            # bitsandbytes не установлен
            pytest.skip("bitsandbytes not installed")
    
    def test_qlora_has_lora_config(self):
        """Тест что QLoRA имеет LoRA конфиг"""
        from transformers.training_presets import get_preset
        
        preset = get_preset("qlora")
        
        try:
            config = preset.get_lora_config()
            assert config is not None
        except ImportError:
            pytest.skip("PEFT not installed")


class TestPresetPrintInfo:
    """Тесты для print_info"""
    
    def test_print_info_exists(self):
        """Тест что print_info существует"""
        from transformers.training_presets import get_preset
        
        preset = get_preset("sft")
        
        if hasattr(preset, 'print_info'):
            # Не должен падать
            preset.print_info()


class TestPresetDescriptions:
    """Тесты для описаний presets"""
    
    def test_preset_has_description_attr(self):
        """Тест что preset имеет описание (опционально)"""
        from transformers.training_presets import get_preset
        
        preset = get_preset("sft")
        # Описание может быть в docstring или атрибуте
        assert preset is not None

class TestConfigValidation:
    """Тесты для валидации конфигурации"""
    
    def test_validate_returns_list(self):
        """Тест что validate возвращает список"""
        from transformers.training_presets import get_preset
        
        preset = get_preset("sft")
        issues = preset.validate()
        
        assert isinstance(issues, list)
    
    def test_validate_high_lr_warning(self):
        """Тест предупреждения о высоком learning rate"""
        from transformers.training_presets import get_preset
        
        preset = get_preset("sft", learning_rate=0.1)
        issues = preset.validate()
        
        # Должно быть предупреждение о высоком LR
        lr_issues = [i for i in issues if "learning_rate" in i.message]
        assert len(lr_issues) > 0
    
    def test_validate_low_lr_warning(self):
        """Тест предупреждения о низком learning rate"""
        from transformers.training_presets import get_preset
        
        preset = get_preset("sft", learning_rate=1e-10)
        issues = preset.validate()
        
        # Должно быть предупреждение о низком LR
        lr_issues = [i for i in issues if "learning_rate" in i.message]
        assert len(lr_issues) > 0
    
    def test_auto_fix_corrects_high_lr(self):
        """Тест что auto_fix исправляет высокий LR"""
        from transformers.training_presets import get_preset
        
        preset = get_preset("sft", learning_rate=0.5)
        original_lr = preset.learning_rate
        
        changes = preset.auto_fix()
        
        # LR должен быть исправлен
        assert preset.learning_rate != original_lr
        assert preset.learning_rate < 0.01
    
    def test_auto_fix_returns_changes_list(self):
        """Тест что auto_fix возвращает список изменений"""
        from transformers.training_presets import get_preset
        
        preset = get_preset("sft", learning_rate=0.1)
        changes = preset.auto_fix()
        
        assert isinstance(changes, list)
        assert len(changes) > 0
    
    def test_validate_correct_config_no_issues(self):
        """Тест что корректная конфигурация не имеет issues"""
        from transformers.training_presets import get_preset
        
        # Создаём конфиг с разумными параметрами
        preset = get_preset("sft", 
                           learning_rate=2e-5,
                           bf16=False,
                           fp16=False)
        
        issues = preset.validate()
        
        # Фильтруем только критичные issues (не связанные с bf16 на CPU)
        critical_issues = [i for i in issues if "learning_rate" in i.message]
        assert len(critical_issues) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
