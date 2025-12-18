"""
Transformers Forge: Пример Training Presets
============================================

Готовые конфигурации для быстрого старта обучения.

Запуск:
    python training_presets_example.py
"""

from transformers import get_preset, list_presets


def main():
    print("=" * 60)
    print("🔨 Transformers Forge: Training Presets Example")
    print("=" * 60)
    
    # =========================================================================
    # 1. Список доступных presets
    # =========================================================================
    
    print("\n📋 Available presets:")
    
    presets = ["sft", "lora", "qlora", "dpo", "memory_efficient"]
    for name in presets:
        preset = get_preset(name)
        print(f"   • {name}: {preset.description}")
    
    # =========================================================================
    # 2. SFT Preset
    # =========================================================================
    
    print("\n" + "=" * 40)
    print("📦 SFT Preset")
    print("=" * 40)
    
    sft = get_preset("sft")
    sft.print_info()
    
    args = sft.get_training_args()
    print(f"\nTrainingArguments preview:")
    print(f"   learning_rate: {args.learning_rate}")
    print(f"   epochs: {args.num_train_epochs}")
    print(f"   batch_size: {args.per_device_train_batch_size}")
    
    # =========================================================================
    # 3. LoRA Preset
    # =========================================================================
    
    print("\n" + "=" * 40)
    print("📦 LoRA Preset")
    print("=" * 40)
    
    lora = get_preset("lora")
    lora.print_info()
    
    lora_config = lora.get_lora_config()
    print(f"\nLoraConfig preview:")
    print(f"   r: {lora_config.r}")
    print(f"   lora_alpha: {lora_config.lora_alpha}")
    print(f"   target_modules: {lora_config.target_modules}")
    
    # =========================================================================
    # 4. QLoRA Preset
    # =========================================================================
    
    print("\n" + "=" * 40)
    print("📦 QLoRA Preset")
    print("=" * 40)
    
    qlora = get_preset("qlora")
    qlora.print_info()
    
    bnb_config = qlora.get_bnb_config()
    print(f"\nBitsAndBytesConfig preview:")
    print(f"   load_in_4bit: {bnb_config.load_in_4bit}")
    print(f"   bnb_4bit_quant_type: {bnb_config.bnb_4bit_quant_type}")
    
    # =========================================================================
    # 5. Кастомизация Preset
    # =========================================================================
    
    print("\n" + "=" * 40)
    print("⚙️ Custom QLoRA Preset")
    print("=" * 40)
    
    custom = get_preset(
        "qlora",
        lora_r=64,
        lora_alpha=128,
        learning_rate=1e-4,
        num_train_epochs=5
    )
    
    print("Overrides applied:")
    print(f"   lora_r: 64 (default: 16)")
    print(f"   lora_alpha: 128 (default: 32)")
    print(f"   learning_rate: 1e-4 (default: 2e-4)")
    print(f"   epochs: 5 (default: 3)")
    
    # =========================================================================
    # 6. Memory Efficient Preset
    # =========================================================================
    
    print("\n" + "=" * 40)
    print("📦 Memory Efficient Preset")
    print("=" * 40)
    
    mem_eff = get_preset("memory_efficient")
    mem_eff.print_info()
    
    print("\nBest for:")
    print("   • Limited GPU memory")
    print("   • Large models on small GPUs")
    print("   • Consumer hardware")
    
    # =========================================================================
    # Итоги
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("🎉 Training Presets Example completed!")
    print("=" * 60)
    print("\nQuick usage:")
    print("  preset = get_preset('qlora')")
    print("  args = preset.get_training_args()")
    print("  lora_config = preset.get_lora_config()")
    print("  bnb_config = preset.get_bnb_config()")


if __name__ == "__main__":
    main()
