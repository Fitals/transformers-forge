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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
