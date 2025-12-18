"""
Transformers Forge: Пример использования EMA
=============================================

EMA (Exponential Moving Average) улучшает качество модели на +1-3%.

Запуск:
    python ema_example.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers.ema import EMACallback, compute_optimal_decay, print_ema_info
from datasets import Dataset


def main():
    print("=" * 60)
    print("🔨 Transformers Forge: EMA Example")
    print("=" * 60)
    
    # =========================================================================
    # 1. Демо-модель (маленькая для примера)
    # =========================================================================
    
    print("\n📥 Loading demo model...")
    
    # Используем маленькую модель для демонстрации
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ Model loaded: {model_name}")
    
    # =========================================================================
    # 2. Демо-датасет
    # =========================================================================
    
    print("\n📚 Creating demo dataset...")
    
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world.",
        "Transformers Forge makes training easier.",
        "EMA improves model quality by 1-3%.",
    ] * 100  # Повторяем для демонстрации
    
    dataset = Dataset.from_dict({"text": texts})
    
    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=128,
            padding="max_length"
        )
    
    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    print(f"✅ Dataset created: {len(dataset)} samples")
    
    # =========================================================================
    # 3. Настройка EMA
    # =========================================================================
    
    print("\n⚙️ Configuring EMA...")
    
    # Автоматический расчёт decay
    total_steps = len(dataset) // 4 * 3  # batch_size=4, epochs=3
    optimal_decay = compute_optimal_decay(total_steps)
    
    print(f"Total steps: {total_steps}")
    print(f"Optimal decay: {optimal_decay:.6f}")
    
    # Информация о EMA конфигурации
    print_ema_info(decay=0.999, total_steps=total_steps)
    
    # Создаём callback
    ema_callback = EMACallback(decay=0.999)
    
    # =========================================================================
    # 4. Обучение
    # =========================================================================
    
    print("\n🚀 Starting training with EMA...")
    
    training_args = TrainingArguments(
        output_dir="./output_ema_example",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        callbacks=[ema_callback]
    )
    
    trainer.train()
    
    # =========================================================================
    # 5. Применение EMA весов
    # =========================================================================
    
    print("\n📊 Applying EMA weights...")
    
    # До применения EMA
    print("Before EMA: Regular training weights")
    
    # Применяем EMA
    original_weights = ema_callback.apply_ema(model)
    print("After EMA: Smoothed weights applied")
    
    # =========================================================================
    # 6. Сохранение
    # =========================================================================
    
    print("\n💾 Saving model with EMA weights...")
    
    model.save_pretrained("./output_ema_example/model_with_ema")
    tokenizer.save_pretrained("./output_ema_example/model_with_ema")
    
    # Сохраняем EMA state отдельно
    ema_state = ema_callback.get_ema_state()
    torch.save(ema_state, "./output_ema_example/ema_state.pt")
    
    print("✅ Model and EMA state saved!")
    
    # =========================================================================
    # 7. Восстановление (демонстрация)
    # =========================================================================
    
    print("\n🔄 Demonstration: Restoring original weights...")
    ema_callback.restore_original(model)
    print("✅ Original weights restored")
    
    print("\n" + "=" * 60)
    print("🎉 EMA Example completed!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  • EMA decay=0.999 is a good default for most training")
    print("  • Call apply_ema() BEFORE saving for best quality")
    print("  • EMA typically improves eval metrics by +1-3%")


if __name__ == "__main__":
    main()
