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
        assert preset.name == "sft"
    
    def test_get_lora_preset(self):
        """Тест получения LoRA preset"""
        from transformers.training_presets import get_preset
        
        preset = get_preset("lora")
        
        assert preset is not None
        assert preset.name == "lora"
    
    def test_get_qlora_preset(self):
        """Тест получения QLoRA preset"""
        from transformers.training_presets import get_preset
        
        preset = get_preset("qlora")
        
        assert preset is not None
        assert preset.name == "qlora"
    
    def test_get_dpo_preset(self):
        """Тест получения DPO preset"""
        from transformers.training_presets import get_preset
        
        preset = get_preset("dpo")
        
        assert preset is not None
        assert preset.name == "dpo"
    
    def test_get_memory_efficient_preset(self):
        """Тест получения memory_efficient preset"""
        from transformers.training_presets import get_preset
        
        preset = get_preset("memory_efficient")
        
        assert preset is not None
        assert preset.name == "memory_efficient"
    
    def test_get_preset_with_overrides(self):
        """Тест получения preset с переопределениями"""
        from transformers.training_presets import get_preset
        
        preset = get_preset("qlora", lora_r=64, learning_rate=1e-4)
        
        assert preset.lora_r == 64
        assert preset.learning_rate == 1e-4
    
    def test_get_unknown_preset(self):
        """Тест получения несуществующего preset"""
        from transformers.training_presets import get_preset
        
        with pytest.raises((ValueError, KeyError)):
            get_preset("unknown_preset")


class TestSFTPreset:
    """Тесты для SFT preset"""
    
    def test_get_training_args(self):
        """Тест получения training args"""
        from transformers.training_presets import get_preset
        
        preset = get_preset("sft")
        args = preset.get_training_args()
        
        # Проверяем что возвращается TrainingArguments
        assert hasattr(args, "learning_rate")
        assert hasattr(args, "num_train_epochs")
        assert hasattr(args, "per_device_train_batch_size")
    
    def test_training_args_values(self):
        """Тест значений training args"""
        from transformers.training_presets import get_preset
        
        preset = get_preset("sft")
        args = preset.get_training_args()
        
        # Проверяем разумные значения
        assert args.learning_rate > 0
        assert args.num_train_epochs > 0
        assert args.per_device_train_batch_size > 0


class TestLoRAPreset:
    """Тесты для LoRA preset"""
    
    def test_get_lora_config(self):
        """Тест получения LoRA конфига"""
        from transformers.training_presets import get_preset
        
        preset = get_preset("lora")
        
        try:
            config = preset.get_lora_config()
            assert hasattr(config, "r")
            assert hasattr(config, "lora_alpha")
        except ImportError:
            # PEFT не установлен
            pytest.skip("PEFT not installed")
    
    def test_lora_r_override(self):
        """Тест переопределения lora_r"""
        from transformers.training_presets import get_preset
        
        preset = get_preset("lora", lora_r=128)
        
        assert preset.lora_r == 128


class TestQLoRAPreset:
    """Тесты для QLoRA preset"""
    
    def test_get_bnb_config(self):
        """Тест получения BitsAndBytes конфига"""
        from transformers.training_presets import get_preset
        
        preset = get_preset("qlora")
        
        try:
            config = preset.get_bnb_config()
            assert hasattr(config, "load_in_4bit")
            assert config.load_in_4bit == True
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
    
    def test_print_info_sft(self, capsys):
        """Тест вывода информации SFT"""
        from transformers.training_presets import get_preset
        
        preset = get_preset("sft")
        preset.print_info()
        
        captured = capsys.readouterr()
        assert "sft" in captured.out.lower() or "SFT" in captured.out
    
    def test_print_info_qlora(self, capsys):
        """Тест вывода информации QLoRA"""
        from transformers.training_presets import get_preset
        
        preset = get_preset("qlora")
        preset.print_info()
        
        captured = capsys.readouterr()
        assert len(captured.out) > 0


class TestPresetDescriptions:
    """Тесты для описаний presets"""
    
    def test_all_presets_have_description(self):
        """Тест что все presets имеют описание"""
        from transformers.training_presets import get_preset
        
        preset_names = ["sft", "lora", "qlora", "dpo", "memory_efficient"]
        
        for name in preset_names:
            preset = get_preset(name)
            assert hasattr(preset, "description")
            assert len(preset.description) > 0


class TestPresetAutoDetection:
    """Тесты для автоопределения параметров"""
    
    def test_auto_detect_device(self):
        """Тест автоопределения устройства"""
        import torch
        from transformers.training_presets import get_preset
        
        preset = get_preset("sft")
        args = preset.get_training_args()
        
        # На CPU не должен использовать GPU-специфичные опции некорректно
        # (проверка что не падает)
        assert args is not None
    
    def test_auto_detect_fp16_bf16(self):
        """Тест автоопределения precision"""
        import torch
        from transformers.training_presets import get_preset
        
        preset = get_preset("sft")
        args = preset.get_training_args()
        
        # Должен определить что-то
        # На CPU может быть False для обоих
        assert hasattr(args, "fp16")
        assert hasattr(args, "bf16")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
