"""
Transformers Forge: Пример заморозки слоёв
==========================================

Layer freezing экономит память и может улучшить качество (LP-LoRA стиль).

Запуск:
    python layer_freezing_example.py
"""

from transformers import AutoModelForCausalLM
from transformers import (
    freeze_first_n_layers,
    freeze_embeddings,
    unfreeze_model,
    get_num_layers,
    get_trainable_params,
    get_frozen_percentage,
    print_layer_status,
    setup_lp_lora_style,
    get_memory_savings_estimate,
    GradualUnfreezer
)


def main():
    print("=" * 60)
    print("🔨 Transformers Forge: Layer Freezing Example")
    print("=" * 60)
    
    # =========================================================================
    # 1. Загрузка модели
    # =========================================================================
    
    print("\n📥 Loading model...")
    
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    num_layers = get_num_layers(model)
    total_params = get_trainable_params(model)
    
    print(f"✅ Model: {model_name}")
    print(f"   Layers: {num_layers}")
    print(f"   Parameters: {total_params:,}")
    
    # =========================================================================
    # 2. Базовый статус
    # =========================================================================
    
    print("\n📊 Initial state:")
    print(f"   Trainable: {get_trainable_params(model):,}")
    print(f"   Frozen: {get_frozen_percentage(model):.1f}%")
    
    # =========================================================================
    # 3. Заморозка первых N слоёв
    # =========================================================================
    
    print("\n🥶 Freezing first 6 layers...")
    
    freeze_first_n_layers(model, n=6)
    
    print(f"   Trainable: {get_trainable_params(model):,}")
    print(f"   Frozen: {get_frozen_percentage(model):.1f}%")
    
    # =========================================================================
    # 4. Детальный статус
    # =========================================================================
    
    print("\n📋 Layer status:")
    print_layer_status(model)
    
    # =========================================================================
    # 5. Разморозка
    # =========================================================================
    
    print("\n🔥 Unfreezing all...")
    
    unfreeze_model(model)
    print(f"   Frozen: {get_frozen_percentage(model):.1f}%")
    
    # =========================================================================
    # 6. LP-LoRA стиль (50% заморозка)
    # =========================================================================
    
    print("\n⚡ Setting up LP-LoRA style (50% frozen)...")
    
    setup_lp_lora_style(model, freeze_ratio=0.5)
    
    print(f"   Trainable: {get_trainable_params(model):,}")
    print(f"   Frozen: {get_frozen_percentage(model):.1f}%")
    
    # =========================================================================
    # 7. Оценка экономии памяти
    # =========================================================================
    
    print("\n💾 Memory savings estimate:")
    
    savings = get_memory_savings_estimate(model)
    print(f"   Gradient memory saved: {savings['gradient_saved_gb']:.4f} GB")
    print(f"   Optimizer memory saved: {savings['optimizer_saved_gb']:.4f} GB")
    print(f"   Total saved: {savings['total_saved_gb']:.4f} GB")
    
    # =========================================================================
    # 8. Постепенная разморозка (демо)
    # =========================================================================
    
    print("\n🔄 Gradual unfreezing demonstration:")
    
    # Сброс
    unfreeze_model(model)
    
    # Создаём unfreezer для 5 эпох
    unfreezer = GradualUnfreezer(model, total_epochs=5, verbose=True)
    
    for epoch in range(5):
        print(f"\n   Epoch {epoch + 1}:")
        unfreezer.step(epoch)
        print(f"   Frozen: {get_frozen_percentage(model):.1f}%")
    
    # =========================================================================
    # Итоги
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("🎉 Layer Freezing Example completed!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  • freeze_first_n_layers() для LP-LoRA стиля")
    print("  • setup_lp_lora_style(model, 0.5) для быстрой настройки")
    print("  • GradualUnfreezer для постепенной разморозки")
    print("  • Экономия памяти пропорциональна % замороженных слоёв")


if __name__ == "__main__":
    main()
